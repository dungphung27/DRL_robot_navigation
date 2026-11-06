#!/usr/bin/env python3
"""Trace an existing DQN's greedy actions, turning oscillations and Q-values.

Copy this file to ~/rl-robot-navigation beside dqn_experiments.py and run:
  python trace_dqn_actions.py --checkpoint checkpoints/.../checkpoint_1000000.pt --seed 2029

It does NOT modify the checkpoint, environment, rewards or training code.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dqn_experiments import close_world, load_checkpoint, make_world, seed_everything

ACTION_NAMES = ('FORWARD', 'LEFT', 'RIGHT')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2029, help='Episode seed, e.g. 2029')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--render', action='store_true', help='Show Tkinter simulation')
    parser.add_argument('--delay', type=float, default=0.0, help='Seconds to pause after every rendered step')
    parser.add_argument('--output', type=Path, default=Path('results/diagnostics'))
    args = parser.parse_args()
    if args.steps < 1 or args.delay < 0:
        parser.error('--steps must be positive and --delay >= 0')

    device = torch.device('cpu')
    model, metadata = load_checkpoint(args.checkpoint, device)
    seed_everything(args.seed)
    world = make_world(render=args.render)
    rows: list[dict] = []
    prev_action = None
    turn_pairs = 0
    turn_reversals = 0
    longest_turn_streak = turn_streak = 0
    longest_alternation = alternation = 0
    best_distance = float('inf')
    outcome = 'TIMEOUT'
    try:
        # Re-seeding immediately before reset matches diagnose_dqn.py.
        seed_everything(args.seed)
        state = world.reset(render=args.render)
        start = tuple(float(x) for x in world.pos)
        goal = tuple(float(x) for x in world.goal)
        initial_distance = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        best_distance = initial_distance
        print(f'Checkpoint: {args.checkpoint}')
        print(f'Seed: {args.seed} | Start: {start} | Goal: {goal}')

        for t in range(args.steps):
            with torch.inference_mode():
                q = model(torch.as_tensor(state, dtype=torch.float32, device=device)).detach().cpu().flatten().numpy()
            if len(q) != 3:
                raise ValueError(f'Expected 3 Q-values, received {len(q)}')
            action = int(np.argmax(q))
            is_turn = action in (1, 2)
            prev_is_turn = prev_action in (1, 2)
            reversal = bool(is_turn and prev_is_turn and action != prev_action)
            if is_turn and prev_is_turn:
                turn_pairs += 1
                if reversal:
                    turn_reversals += 1
            if is_turn:
                turn_streak += 1
                alternation = (alternation + 1) if reversal else 1
            else:
                turn_streak = 0
                alternation = 0
            longest_turn_streak = max(longest_turn_streak, turn_streak)
            longest_alternation = max(longest_alternation, alternation)

            old_pos = tuple(float(v) for v in world.pos)
            old_angle = float(world.angle)
            relative_goal_angle = float(state[2])
            next_state, reward, terminated = world.step(action, render=args.render)
            new_pos = tuple(float(v) for v in world.pos)
            moved = float(np.hypot(new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]))
            distance = float(np.hypot(goal[0] - new_pos[0], goal[1] - new_pos[1]))
            best_distance = min(best_distance, distance)
            rows.append({
                'step': t + 1, 'action': ACTION_NAMES[action], 'action_index': action,
                'x': round(new_pos[0], 6), 'y': round(new_pos[1], 6),
                'angle_before_deg': round(old_angle, 3),
                'angle_after_deg': round(float(world.angle), 3),
                'goal_angle_relative_before_deg': round(relative_goal_angle, 3),
                'goal_distance': round(distance, 3),
                'best_goal_distance': round(best_distance, 3),
                'moved': round(moved, 3), 'reward': round(float(reward), 5),
                'q_forward': round(float(q[0]), 6),
                'q_left': round(float(q[1]), 6),
                'q_right': round(float(q[2]), 6),
                'q_best_margin': round(float(np.sort(q)[-1] - np.sort(q)[-2]), 6),
                'turn_reversal': int(reversal),
                'consecutive_turn_steps': turn_streak,
                'consecutive_alternating_turn_steps': alternation,
            })
            state = next_state
            prev_action = action
            if args.render and args.delay:
                time.sleep(args.delay)
            if terminated:
                outcome = 'COLLISION' if world.has_collision(world.pos) else 'SUCCESS'
                break
    finally:
        close_world(world)

    folder = args.output / ('trace_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    csv_path = folder / ('trace_seed_' + str(args.seed) + '.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    actions = [r['action_index'] for r in rows]
    counts = {name: actions.count(i) for i, name in enumerate(ACTION_NAMES)}
    switch_rate = 100 * turn_reversals / turn_pairs if turn_pairs else 0.0
    total_movement = sum(r['moved'] for r in rows)
    print(f'Outcome: {outcome} | Steps: {len(rows)} | Actions: {counts}')
    print(f'Path length: {total_movement:.1f} | Best goal distance: {best_distance:.1f} '
          f'(started at {initial_distance:.1f})')
    print(f'Turn-only streak (longest): {longest_turn_streak} steps')
    print(f'LEFT<->RIGHT reversals: {turn_reversals} / {turn_pairs} adjacent turn pairs '
          f'({switch_rate:.1f}%)')
    print(f'Longest strict alternating-turn streak: {longest_alternation} steps')
    print('First 40 actions:', ' '.join('F' if a == 0 else 'L' if a == 1 else 'R' for a in actions[:40]))
    print('Saved per-step trace:', csv_path)
    print('Inspect consecutive rows and q_left/q_right when the robot turns repeatedly.')


if __name__ == '__main__':
    main()
