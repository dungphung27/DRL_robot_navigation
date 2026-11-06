#!/usr/bin/env python3
"""Reproducible DQN navigation evaluation and architecture experiments.

Place beside dqn.py, envs/, exp_schedule.py and report_plots.py.
Compatible with the DQN state-dict and custom checkpoint formats used in
this conversation. Does not overwrite the original dqn.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from dqn import DQN
from envs.obstacles_env import ObstaclesWorld
from exp_schedule import ExponentialSchedule


def stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def make_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_world(render: bool = True):
    return ObstaclesWorld(300, 300, see_all=render)


def close_world(world):
    try:
        world.top.destroy()
    except Exception:
        pass


def load_checkpoint(path: Path, device: torch.device):
    """Read .pth state_dict or .pt checkpoint, infer MLP depth and dimensions."""
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    raw = torch.load(path, map_location='cpu', weights_only=True)
    state = raw
    metadata = {}
    if isinstance(raw, dict):
        metadata = {k: raw[k] for k in ('step', 'epoch', 'num_steps') if k in raw}
        if isinstance(raw.get('model'), dict):
            model_blob = raw['model']
            state = model_blob.get('state_dict', model_blob)
        else:
            state = raw.get('model_state_dict', raw.get('state_dict', raw))
    if not isinstance(state, dict):
        raise ValueError(f'Unsupported checkpoint layout: {path}')

    # The user's DQN stores nn.Linear parameters as linear_stack.0.weight,
    # linear_stack.2.weight, etc. Count these to infer hidden layer count.
    linear_weights = [(int(k.split('.')[1]), v) for k, v in state.items()
                      if k.startswith('linear_stack.') and k.endswith('.weight')
                      and k.split('.')[1].isdigit() and getattr(v, 'ndim', None) == 2]
    linear_weights.sort(key=lambda x: x[0])
    if len(linear_weights) < 2:
        raise ValueError('Checkpoint does not appear to use the expected DQN linear_stack MLP.')
    layer_dims = [int(linear_weights[0][1].shape[1])] + [int(t.shape[0]) for _, t in linear_weights]
    input_dim, output_dim = layer_dims[0], layer_dims[-1]
    hidden_dim = layer_dims[1]
    if any(d != hidden_dim for d in layer_dims[1:-1]):
        raise ValueError('Mixed hidden sizes are not supported by the original DQN class.')
    model = DQN(input_dim, output_dim, num_layers=len(linear_weights), hidden_dim=hidden_dim)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.device = device  # Original DQN.forward refers to self.device.
    metadata.update({'checkpoint': str(path), 'input_dim': input_dim,
                     'output_dim': output_dim, 'hidden_layers': len(linear_weights) - 1,
                     'hidden_dim': hidden_dim})
    return model, metadata


def evaluate_policy(model, *, episodes: int, seed: int, max_steps: int = 1000,
                    render: bool = True):
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
    else:
        device = torch.device('cpu')
    seed_everything(seed)
    world = make_world(render=render)
    rows = []
    try:
        for ep in range(episodes):
            # Identical episode initializations across all candidate models.
            # The constructor and subsequent trajectories cannot change this
            # episode's reset random state.
            episode_seed = seed + ep
            seed_everything(episode_seed)
            state = world.reset(render=render)
            start = tuple(float(x) for x in world.pos)
            goal = tuple(float(x) for x in world.goal)
            if model is not None and state.shape[0] != model.state_dim:
                raise ValueError(f'Observation has {state.shape[0]} features but model expects '
                                 f'{model.state_dim}. Check generic_env.py and checkpoint.')
            total_reward = 0.0
            discounted_return = 0.0
            path_length = 0.0
            outcome = 'TIMEOUT'
            for step in range(max_steps):
                previous_pos = world.pos
                if model is None:  # explicitly requested random-policy baseline
                    action = int(np.random.randint(3))
                else:
                    with torch.no_grad():
                        q = model(torch.as_tensor(state, dtype=torch.float32, device=device))
                        action = int(q.argmax().item())
                state, reward, terminated = world.step(action, render=render)
                path_length += math.dist(previous_pos, world.pos)
                total_reward += float(reward)
                discounted_return += (0.99 ** step) * float(reward)
                if terminated:
                    outcome = 'SUCCESS' if world._within_goal() and not world.has_collision(world.pos) else 'COLLISION'
                    break
            rows.append({'episode': ep, 'seed': episode_seed, 'start_x': start[0],
                         'start_y': start[1], 'goal_x': goal[0], 'goal_y': goal[1],
                         'outcome': outcome, 'steps': step + 1,
                         'total_reward': total_reward, 'discounted_return': discounted_return,
                         'path_length': path_length,
                         'steps_to_success': step + 1 if outcome == 'SUCCESS' else ''})
    finally:
        close_world(world)
    return rows


def summarize(rows: list[dict]):
    n = len(rows)
    counts = Counter(r['outcome'] for r in rows)
    success_steps = [r['steps'] for r in rows if r['outcome'] == 'SUCCESS']
    return {'episodes': n,
            'success': counts['SUCCESS'], 'collision': counts['COLLISION'],
            'timeout': counts['TIMEOUT'],
            'success_rate_pct': 100 * counts['SUCCESS'] / n,
            'collision_rate_pct': 100 * counts['COLLISION'] / n,
            'timeout_rate_pct': 100 * counts['TIMEOUT'] / n,
            'mean_steps_to_success': float(np.mean(success_steps)) if success_steps else None,
            'median_steps_to_success': float(np.median(success_steps)) if success_steps else None,
            'mean_total_reward': float(np.mean([r['total_reward'] for r in rows])),
            'mean_path_length_success': (float(np.mean([r['path_length'] for r in rows
                                                       if r['outcome'] == 'SUCCESS']))
                                         if success_steps else None)}


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_charts(summary_rows: list[dict], folder: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not summary_rows:
        return
    labels = [str(r['name']) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.7), 5))
    x = np.arange(len(labels))
    w = 0.26
    for offset, key, label in [(-w, 'success_rate_pct', 'Success'),
                               (0, 'collision_rate_pct', 'Collision'),
                               (w, 'timeout_rate_pct', 'Timeout')]:
        ax.bar(x + offset, [r[key] for r in summary_rows], w, label=label)
    ax.set_xticks(x, labels, rotation=25, ha='right')
    ax.set_ylabel('Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title('DQN Evaluation: Episode Outcome Rates (Greedy Policy)')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'evaluation_rates.{ext}', dpi=250)
    plt.close(fig)


def command_evaluate(args):
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    root = make_dir(Path(args.output) / f'evaluation_{stamp()}')
    checkpoints = [Path(s) for s in (args.checkpoints or [])]
    if args.random_policy:
        checkpoints.append(None)
    if not checkpoints:
        raise ValueError('Supply --checkpoints PATH... and/or --random-policy')
    results = []
    for path in checkpoints:
        if path is None:
            model, meta, name = None, {'checkpoint': 'random_policy'}, 'Random'
        else:
            model, meta = load_checkpoint(path, device)
            name = path.parent.name + '/' + path.stem
        policy_desc = 'random actions' if model is None else 'epsilon=0'
        print(f'\nEvaluating {name}: {args.episodes} episodes, {policy_desc}')
        rows = evaluate_policy(model, episodes=args.episodes, seed=args.seed,
                               max_steps=args.max_steps, render=args.render)
        label = f'model_{len(results):02d}'
        save_csv(root / f'{label}_episodes.csv', rows)
        score = summarize(rows)
        score.update({'name': name, 'checkpoint': meta['checkpoint']})
        results.append(score)
        (root / f'{label}_model_info.json').write_text(json.dumps(meta, indent=2, default=str))
        print(f"Success {score['success_rate_pct']:.1f}% | Collision {score['collision_rate_pct']:.1f}% "
              f"| Timeout {score['timeout_rate_pct']:.1f}%")
    save_csv(root / 'comparison.csv', results)
    save_summary_charts(results, root)
    (root / 'evaluation_config.json').write_text(json.dumps(vars(args), indent=2, default=str))
    print(f'\nEvaluation results saved to: {root}')


def train_one(args, *, hidden_layers: int, algorithm: str, seed: int, batch_root: Path):
    seed_everything(seed)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    tag = f'{algorithm}_{hidden_layers}H_seed{seed}'
    run_root = make_dir(batch_root / tag)
    ckpt_dir = make_dir(run_root / 'checkpoints')
    report_dir = make_dir(run_root / 'report')
    cfg = {'algorithm': algorithm, 'hidden_layers': hidden_layers,
           'hidden_dim': args.hidden_dim, 'steps': args.steps, 'seed': seed,
           'learning_rate': args.lr, 'gamma': args.gamma,
           'batch_size': args.batch_size, 'replay_size': args.replay_size,
           'target_update_steps': args.target_update,
           'gradient_update_steps': args.train_every,
           'epsilon_start': 1.0, 'epsilon_end': 0.01,
           'epsilon_decay_steps': args.epsilon_decay,
           'max_episode_steps': args.max_steps,
           'timeout_handling': 'truncation_bootstraps_in_TD_target',
           'state_dim': 64, 'actions': 3}
    (run_root / 'config.json').write_text(json.dumps(cfg, indent=2))

    world = make_world(render=True)
    online = DQN(64, 3, num_layers=hidden_layers + 1, hidden_dim=args.hidden_dim).to(device)
    target = DQN(64, 3, num_layers=hidden_layers + 1, hidden_dim=args.hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    online.device = device
    target.device = device
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    exploration = ExponentialSchedule(1.0, 0.01, args.epsilon_decay)
    memory = deque(maxlen=args.replay_size)
    state = world.reset(render=True)
    if state.shape != (64,):
        close_world(world)
        raise ValueError(f'Expected 64-dimensional state, got {state.shape}')
    ep_rewards = []
    episode_returns, episode_totals, episode_lengths, outcomes, losses = [], [], [], [], []
    ep_step = 0
    try:
        pbar = trange(args.steps, desc=tag, unit='step')
        for t in pbar:
            eps = exploration.value(t)
            if np.random.random() < eps:
                action = int(np.random.randint(3))
            else:
                with torch.no_grad():
                    value = online(torch.as_tensor(state, dtype=torch.float32, device=device))
                    action = int(value.argmax().item())
            next_state, reward, terminated = world.step(action, render=True)
            ep_step += 1
            timeout = ep_step >= args.max_steps and not terminated
            finished = bool(terminated or timeout)
            # Store actual environment termination, NOT timeout/truncation.
            memory.append((np.asarray(state, dtype=np.float32).copy(), action,
                           float(reward), np.asarray(next_state, dtype=np.float32).copy(),
                           bool(terminated)))
            ep_rewards.append(float(reward))
            state = next_state

            if (t + 1) % args.train_every == 0 and len(memory) >= args.batch_size:
                batch = random.sample(memory, args.batch_size)
                s = torch.as_tensor(np.stack([b[0] for b in batch]), device=device)
                a = torch.as_tensor([b[1] for b in batch], device=device).long().unsqueeze(1)
                r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                ns = torch.as_tensor(np.stack([b[3] for b in batch]), device=device)
                done = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=device)
                q = online(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    if algorithm == 'double_dqn':
                        best_actions = online(ns).argmax(dim=1, keepdim=True)
                        q_next = target(ns).gather(1, best_actions).squeeze(1)
                    else:
                        q_next = target(ns).max(dim=1).values
                    y = r + args.gamma * (1 - done) * q_next
                loss = F.mse_loss(q, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if (t + 1) % args.target_update == 0:
                target.load_state_dict(online.state_dict())

            if finished:
                # Correct forward discounted return G0 = r0 + gamma*r1 + ...
                disc, power = 0.0, 1.0
                for rr in ep_rewards:
                    disc += power * rr
                    power *= args.gamma
                episode_returns.append(disc)
                episode_totals.append(float(sum(ep_rewards)))
                episode_lengths.append(ep_step)
                outcomes.append('SUCCESS' if terminated and world._within_goal()
                                and not world.has_collision(world.pos)
                                else 'COLLISION' if terminated else 'TIMEOUT')
                ep_rewards.clear()
                ep_step = 0
                state = world.reset(render=True)
                if len(outcomes) % 25 == 0:
                    pbar.set_postfix(episodes=len(outcomes), success=f'{outcomes.count("SUCCESS")/len(outcomes):.1%}')

            if (t + 1) % args.save_every == 0 or t + 1 == args.steps:
                ckpt = {'step': t + 1, 'model': online.custom_dump(),
                        'target_state_dict': target.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': cfg}
                torch.save(ckpt, ckpt_dir / f'checkpoint_{t+1}.pt')
    finally:
        close_world(world)

    # Also save plain state_dict, matching the existing dqn_100_0.pth format.
    torch.save(online.state_dict(), ckpt_dir / 'dqn_final.pth')
    from report_plots import export_training_report
    export_training_report(np.array(episode_returns), np.array(episode_lengths),
                           np.array(losses), outcomes, folder=str(report_dir),
                           total_rewards=np.array(episode_totals), window=50)
    eval_rows = evaluate_policy(online, episodes=args.eval_episodes, seed=args.eval_seed,
                                max_steps=args.max_steps, render=True)
    save_csv(run_root / 'evaluation_episodes.csv', eval_rows)
    outcome = summarize(eval_rows)
    outcome.update({'name': tag, 'algorithm': algorithm, 'hidden_layers': hidden_layers,
                    'hidden_dim': args.hidden_dim, 'seed': seed,
                    'checkpoint': str(ckpt_dir / 'dqn_final.pth')})
    (run_root / 'evaluation_summary.json').write_text(json.dumps(outcome, indent=2))
    print(f'{tag}: {outcome["success_rate_pct"]:.1f}% evaluation success')
    return outcome


def save_seed_aggregate(results: list[dict], root: Path):
    """Aggregate repeated seeds without masking the underlying per-run results."""
    groups = defaultdict(list)
    for r in results:
        groups[(r['algorithm'], r['hidden_layers'], r['hidden_dim'])].append(r)
    rows = []
    for (alg, layers, width), cases in sorted(groups.items()):
        rates = [r['success_rate_pct'] for r in cases]
        rows.append({'algorithm': alg, 'hidden_layers': layers, 'hidden_dim': width,
                     'seeds_completed': len(cases),
                     'mean_success_rate_pct': float(np.mean(rates)),
                     'std_success_rate_pct': (float(np.std(rates, ddof=1))
                                              if len(rates) > 1 else None),
                     'mean_collision_rate_pct': float(np.mean([r['collision_rate_pct'] for r in cases])),
                     'mean_timeout_rate_pct': float(np.mean([r['timeout_rate_pct'] for r in cases]))})
    save_csv(root / 'aggregate_by_architecture.csv', rows)


def command_train(args):
    root = make_dir(Path(args.output) / f'experiments_{stamp()}')
    results = []
    for algorithm in args.algorithms:
        for hidden_layers in args.hidden_layers:
            for seed in args.seeds:
                results.append(train_one(args, algorithm=algorithm,
                                         hidden_layers=hidden_layers, seed=seed,
                                         batch_root=root))
                save_csv(root / 'comparison.csv', results)  # progress is durable
                save_seed_aggregate(results, root)
                save_summary_charts(results, root)
    print(f'\nAll experiment results saved to: {root}')


def get_parser():
    p = argparse.ArgumentParser(description='Evaluate existing DQN or train reproducible variants')
    sub = p.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--max-steps', type=int, default=1000, help='Max environment steps per episode')
    common.add_argument('--output', default='results/experiments', help='Parent output directory')
    common.add_argument('--cuda', action='store_true', help='Use CUDA when available')

    ev = sub.add_parser('evaluate', parents=[common], help='Evaluate pretrained model(s), epsilon=0')
    ev.add_argument('--checkpoints', nargs='+', help='One or more .pth/.pt checkpoint paths')
    ev.add_argument('--random-policy', action='store_true', help='Also evaluate random actions')
    ev.add_argument('--episodes', type=int, default=200)
    ev.add_argument('--seed', type=int, default=2026)
    ev.add_argument('--render', action='store_true', help='Show Tk window (use few episodes)')

    tr = sub.add_parser('train', parents=[common], help='Train and evaluate architecture variants')
    tr.add_argument('--hidden-layers', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2, 3])
    tr.add_argument('--algorithms', nargs='+', choices=['dqn', 'double_dqn'], default=['dqn'])
    tr.add_argument('--seeds', type=int, nargs='+', default=[42])
    tr.add_argument('--steps', type=int, default=1_000_000)
    tr.add_argument('--hidden-dim', type=int, default=256)
    tr.add_argument('--lr', type=float, default=1e-3)
    tr.add_argument('--gamma', type=float, default=0.99)
    tr.add_argument('--epsilon-decay', type=int, default=400_000)
    tr.add_argument('--batch-size', type=int, default=64)
    tr.add_argument('--replay-size', type=int, default=30_000)
    tr.add_argument('--train-every', type=int, default=4)
    tr.add_argument('--target-update', type=int, default=10_000)
    tr.add_argument('--save-every', type=int, default=100_000)
    tr.add_argument('--eval-episodes', type=int, default=200)
    tr.add_argument('--eval-seed', type=int, default=2026)
    return p


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.command == 'evaluate':
        if args.episodes <= 0 or args.max_steps <= 0:
            parser.error('--episodes and --max-steps must be positive')
        command_evaluate(args)
    else:
        for name in ('steps', 'batch_size', 'replay_size', 'train_every', 'target_update',
                     'save_every', 'eval_episodes', 'epsilon_decay'):
            if getattr(args, name) <= 0:
                parser.error(f'--{name.replace("_", "-")} must be positive')
        if args.batch_size > args.replay_size:
            parser.error('--batch-size cannot exceed --replay-size')
        command_train(args)


if __name__ == '__main__':
    main()
