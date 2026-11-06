#!/usr/bin/env python3
"""Diagnose DQN navigation failures and paired baseline-vs-new outcomes.

Run from ~/rl-robot-navigation beside dqn_experiments.py.
No training or checkpoint changes. All models use epsilon=0 and the same
episode seeds. Unlike the basic evaluator, measures movement/action stagnation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dqn_experiments import (close_world, load_checkpoint, make_world,
                             seed_everything)


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_with_diagnostics(model, episodes: int, seed: int, max_steps: int,
                              render: bool = False):
    device = next(model.parameters()).device
    seed_everything(seed)
    world = make_world(render=render)
    results = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            seed_everything(episode_seed)
            state = world.reset(render=render)
            start = tuple(float(v) for v in world.pos)
            goal = tuple(float(v) for v in world.goal)
            initial_distance = math.dist(start, goal)
            if state.shape[0] != model.state_dim:
                raise ValueError('State dimension incompatible with checkpoint')

            action_counts = Counter()
            longest_turn_only = 0
            turn_only_streak = 0
            longest_no_goal_progress = 0
            no_goal_progress_streak = 0
            best_distance = initial_distance
            total_path_length = 0.0
            total_reward = 0.0
            last_position = start
            last_distance = initial_distance
            motion_steps = 0
            outcome = 'TIMEOUT'
            step = -1
            for step in range(max_steps):
                with torch.no_grad():
                    q_values = model(torch.as_tensor(state, dtype=torch.float32,
                                                     device=device))
                    action = int(q_values.argmax().item())
                action_counts[action] += 1
                state, reward, terminated = world.step(action, render=render)
                total_reward += float(reward)
                now = tuple(float(v) for v in world.pos)
                displacement = math.dist(last_position, now)
                total_path_length += displacement
                if displacement > 1e-7:
                    motion_steps += 1
                last_position = now

                # Track a contiguous run without forward movement. This
                # reliably detects indefinite rotation with the current actions.
                if action != 0:
                    turn_only_streak += 1
                else:
                    turn_only_streak = 0
                longest_turn_only = max(longest_turn_only, turn_only_streak)

                distance = math.dist(now, goal)
                if distance < best_distance - 0.5:
                    best_distance = distance
                    no_goal_progress_streak = 0
                else:
                    no_goal_progress_streak += 1
                longest_no_goal_progress = max(longest_no_goal_progress,
                                               no_goal_progress_streak)
                last_distance = distance

                if terminated:
                    outcome = ('COLLISION' if world.has_collision(world.pos)
                               else 'SUCCESS' if world._within_goal()
                               else 'OTHER_TERMINATION')
                    break
            steps = step + 1
            forward_ratio = action_counts[0] / steps
            row = {
                'episode': episode, 'seed': episode_seed,
                'start_x': start[0], 'start_y': start[1],
                'goal_x': goal[0], 'goal_y': goal[1],
                'outcome': outcome, 'steps': steps,
                'forward': action_counts[0],
                'left': action_counts[1], 'right': action_counts[2],
                'forward_pct': round(100 * forward_ratio, 2),
                'max_consecutive_turn_steps': longest_turn_only,
                'max_steps_without_new_goal_progress': longest_no_goal_progress,
                'steps_with_movement': motion_steps,
                'path_length': round(total_path_length, 3),
                'start_goal_distance': round(initial_distance, 3),
                'end_goal_distance': round(last_distance, 3),
                'net_goal_progress': round(initial_distance-last_distance, 3),
                'total_reward': round(total_reward, 3),
            }
            results.append(row)
            if render:
                print(f"Episode {episode:3d} | {outcome:7s} | "
                      f"F/L/R={action_counts[0]}/{action_counts[1]}/{action_counts[2]} | "
                      f"max-turn={longest_turn_only} | "
                      f"goal progress={row['net_goal_progress']:.1f}")
    finally:
        close_world(world)
    return results


def summarize(rows):
    n = len(rows)
    counts = Counter(r['outcome'] for r in rows)
    timeouts = [r for r in rows if r['outcome'] == 'TIMEOUT']
    def frac(yes, denom):
        return round(100 * yes / denom, 2) if denom else 0.0
    return {
        'episodes': n,
        'success_rate_pct': frac(counts['SUCCESS'], n),
        'collision_rate_pct': frac(counts['COLLISION'], n),
        'timeout_rate_pct': frac(counts['TIMEOUT'], n),
        # Thresholds are descriptive heuristics, not objective proof of cause.
        'timeout_with_turn_streak_ge_100': sum(
            r['max_consecutive_turn_steps'] >= 100 for r in timeouts),
        'timeout_with_forward_pct_lt_10': sum(
            r['forward_pct'] < 10 for r in timeouts),
        'timeout_with_no_new_best_goal_distance_ge_200': sum(
            r['max_steps_without_new_goal_progress'] >= 200 for r in timeouts),
        'timeout_mean_forward_pct': round(float(np.mean(
            [r['forward_pct'] for r in timeouts])), 2) if timeouts else None,
        'timeout_mean_path_length': round(float(np.mean(
            [r['path_length'] for r in timeouts])), 2) if timeouts else None,
    }


def paired_outcomes(rows_a, rows_b):
    a, b = {r['seed']: r for r in rows_a}, {r['seed']: r for r in rows_b}
    if a.keys() != b.keys():
        raise ValueError('Models were evaluated on different episode seeds')
    paired = []
    counts = Counter()
    for episode_seed in sorted(a):
        x, y = a[episode_seed], b[episode_seed]
        change = f"{x['outcome']} -> {y['outcome']}"
        counts[change] += 1
        paired.append({
            'seed': episode_seed, 'old_outcome': x['outcome'],
            'new_outcome': y['outcome'],
            'old_forward_pct': x['forward_pct'], 'new_forward_pct': y['forward_pct'],
            'old_max_turn_streak': x['max_consecutive_turn_steps'],
            'new_max_turn_streak': y['max_consecutive_turn_steps'],
            'old_goal_progress': x['net_goal_progress'],
            'new_goal_progress': y['net_goal_progress'],
            'outcome_change': change,
        })
    return paired, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', default='results/diagnostics')
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('--episodes and --max-steps must be positive')
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    folder = Path(args.output) / ('diagnose_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    summaries, model_rows = [], []
    for idx, filename in enumerate(args.checkpoints):
        model, meta = load_checkpoint(Path(filename), device)
        print(f'\nEvaluating [{idx}] {filename} (epsilon=0, {args.episodes} episodes)...', flush=True)
        rows = evaluate_with_diagnostics(model, args.episodes, args.seed,
                                         args.max_steps, args.render)
        model_rows.append(rows)
        save_csv(folder / f'model_{idx:02d}_diagnostics.csv', rows)
        summary = summarize(rows)
        summary.update({'model_id': idx, 'checkpoint': filename})
        summaries.append(summary)
        print(f"Success {summary['success_rate_pct']}% | "
              f"Collision {summary['collision_rate_pct']}% | "
              f"Timeout {summary['timeout_rate_pct']}%")
        print(f"Of {sum(r['outcome']=='TIMEOUT' for r in rows)} timeouts: "
              f"{summary['timeout_with_turn_streak_ge_100']} have a >=100-step "
              f"turn-only streak; {summary['timeout_with_forward_pct_lt_10']} "
              f"have <10% FORWARD actions; "
              f"{summary['timeout_with_no_new_best_goal_distance_ge_200']} "
              f"have >=200 steps without reaching a new best goal distance.")
    save_csv(folder / 'diagnostic_summary.csv', summaries)
    if len(model_rows) == 2:
        paired, transitions = paired_outcomes(model_rows[0], model_rows[1])
        save_csv(folder / 'paired_episode_outcomes.csv', paired)
        print('\nPaired episode changes (model 0 -> model 1):')
        for change, count in transitions.most_common():
            print(f'  {change}: {count}')
        gained = [r['seed'] for r in paired if
                  r['old_outcome'] != 'SUCCESS' and r['new_outcome'] == 'SUCCESS']
        lost = [r['seed'] for r in paired if
                r['old_outcome'] == 'SUCCESS' and r['new_outcome'] != 'SUCCESS']
        print('Sample seeds gained success:', gained[:12])
        print('Sample seeds lost success:  ', lost[:12])
    (folder / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print('\nSaved:', folder)


if __name__ == '__main__':
    main()
