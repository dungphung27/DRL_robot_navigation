"""Export report-ready training figures and episode metrics for robot DQN."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Generate files without opening GUI windows.
import matplotlib.pyplot as plt


def moving_average(values, window=50):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    w = min(window, values.size)
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='valid')


def export_training_report(returns, lengths, losses, outcomes, folder='results/report', window=50, total_rewards=None):
    """Save PNG/PDF charts plus CSV data. outcomes per completed episode."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    returns = np.asarray(returns, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    losses = np.asarray(losses, dtype=float)
    outcomes = list(outcomes)
    if total_rewards is not None:
        total_rewards = np.asarray(total_rewards, dtype=float)
        if len(total_rewards) != len(returns):
            raise ValueError('total_rewards must have one value per completed episode')
    assert len(returns) == len(lengths) == len(outcomes), (
        f'Lengths mismatch: returns={len(returns)}, lengths={len(lengths)}, outcomes={len(outcomes)}'
    )

    with (folder / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'discounted_return', 'total_reward', 'length', 'outcome'])
        for i, (ret, length, outcome) in enumerate(zip(returns, lengths, outcomes)):
            total = '' if total_rewards is None else float(total_rewards[i])
            writer.writerow([i, float(ret), total, int(length), outcome])
    with (folder / 'losses.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['update_index', 'loss'])
        writer.writerows((i, float(v)) for i, v in enumerate(losses))

    def save_figure(name):
        plt.tight_layout()
        plt.savefig(folder / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(folder / f'{name}.pdf', bbox_inches='tight')
        plt.close()

    def series_chart(values, title, ylabel, name, xlabel, avg_window=window):
        if len(values) == 0:
            return
        plt.figure(figsize=(9, 4.7))
        plt.plot(np.arange(len(values)), values, alpha=0.23, linewidth=0.8, label='Raw')
        smooth = moving_average(values, avg_window)
        start = min(avg_window, len(values)) - 1
        plt.plot(np.arange(start, start + len(smooth)), smooth,
                 linewidth=2, label=f'Moving average ({min(avg_window,len(values))})')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure(name)

    series_chart(returns, 'DQN Training: Discounted Return', 'Discounted return',
                 '01_return', 'Episode')
    series_chart(losses, 'DQN Training: TD Loss', 'MSE loss',
                 '02_loss', 'Gradient update', avg_window=500)
    series_chart(lengths, 'DQN Training: Episode Length', 'Steps',
                 '03_episode_length', 'Episode')

    if total_rewards is not None:
        series_chart(total_rewards, 'DQN Training: Total Reward per Episode',
                     'Undiscounted reward sum', '07_total_reward', 'Episode')
        series_chart(np.cumsum(total_rewards), 'DQN Training: Cumulative Total Reward',
                     'Cumulative reward', '08_cumulative_reward', 'Episode')

    if outcomes:
        indicators = {
            'SUCCESS': np.array([v == 'SUCCESS' for v in outcomes], dtype=float),
            'COLLISION': np.array([v == 'COLLISION' for v in outcomes], dtype=float),
            'TIMEOUT': np.array([v == 'TIMEOUT' for v in outcomes], dtype=float),
        }
        w = min(window, len(outcomes))
        xs = np.arange(w - 1, len(outcomes))
        plt.figure(figsize=(9, 4.7))
        plt.plot(xs, moving_average(indicators['SUCCESS'], w) * 100, linewidth=2)
        plt.title('DQN Training: Rolling Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success rate (%)')
        plt.ylim(0, 100)
        plt.grid(alpha=0.25)
        save_figure('04_success_rate')

        plt.figure(figsize=(9, 4.7))
        for key in ('COLLISION', 'TIMEOUT'):
            plt.plot(xs, moving_average(indicators[key], w) * 100,
                     linewidth=2, label=key.title())
        plt.title('DQN Training: Rolling Failure Rates')
        plt.xlabel('Episode')
        plt.ylabel('Rate (%)')
        plt.ylim(0, 100)
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure('05_failure_rates')

        counts = [int(indicators[key].sum()) for key in indicators]
        plt.figure(figsize=(8, 4.7))
        bars = plt.bar(list(indicators), counts)
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     str(count), ha='center', va='bottom')
        plt.title('DQN Training: Episode Outcomes')
        plt.ylabel('Completed episodes')
        plt.grid(axis='y', alpha=0.25)
        save_figure('06_outcome_counts')

    print(f'Report charts and CSV saved to: {folder.resolve()}')


if __name__ == '__main__':
    # Small smoke test using illustrative data, not a real training result.
    rng = np.random.default_rng(0)
    n = 150
    outcomes = rng.choice(['SUCCESS', 'COLLISION', 'TIMEOUT'], size=n).tolist()
    export_training_report(rng.normal(size=n), rng.integers(1, 1000, n),
                           rng.uniform(size=500), outcomes, folder='results/report_test',
                           total_rewards=rng.normal(20, 60, n))