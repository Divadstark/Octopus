import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import re

# Remove all LaTeX-related settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
})

base_dir = '/results'
agent_dirs = {
    'MDQN': os.path.join(base_dir, 'MDQN'),
    'Rainbow': os.path.join(base_dir, 'Rainbow'),
    'MIQN*': os.path.join(base_dir, 'MIQN*'),
    'Octopus': os.path.join(base_dir, 'Octopus'),
    'C51': os.path.join(base_dir, 'C51'),
    'Double DQN': os.path.join(base_dir, 'Double_Q'),
    'Prioritized': os.path.join(base_dir, 'Prioritized'),
    "QRDQN": os.path.join(base_dir, 'qrdqn')
}

agents_to_include = ['MDQN', 'Rainbow', 'MIQN*', 'C51', 'Octopus', 'Double DQN', 'Prioritized', "QRDQN"]

agent_colors = {
    'MDQN': 'green',
    'Rainbow': 'purple',
    'Octopus': 'red',
    'MIQN*': 'blue',
    'C51': 'orange',
    'Double DQN': 'cyan',
    'Prioritized': 'brown',
    "QRDQN": 'pink'
}


def extract_game_and_seed(filename):
    """Extract game name and seed number from filename."""
    filename_lower = filename.lower()

    seed_match = re.search(r'seed(\d+)', filename_lower)
    seed = int(seed_match.group(1)) if seed_match else None

    game_name = filename_lower
    if seed_match:
        game_name = game_name.replace(seed_match.group(0), '')
    game_name = game_name.replace('.csv', '').strip('_').strip()

    return game_name, seed


def organize_files_by_game(agent_dir):
    """Organize files by game, grouping seeds together."""
    files = [f for f in os.listdir(agent_dir) if f.endswith('.csv')]
    game_files = {}

    for file in files:
        game_name, seed = extract_game_and_seed(file)
        if game_name not in game_files:
            game_files[game_name] = []
        game_files[game_name].append((file, seed))

    return game_files


def check_files_coverage(agents_to_include, agent_dirs):
    print("=== FILE COVERAGE CHECK ===")

    for agent in agents_to_include:
        agent_dir = agent_dirs[agent]
        if not os.path.exists(agent_dir):
            print(f"{agent}: Directory does not exist - {agent_dir}")
            continue

        game_files = organize_files_by_game(agent_dir)
        total_files = sum(len(files) for files in game_files.values())
        print(f"\n{agent}: {total_files} total CSV files, {len(game_files)} unique games")

        for game, files in sorted(game_files.items()):
            seeds = [seed for _, seed in files if seed is not None]
            if seeds:
                print(f"  {game}: {len(files)} files, seeds: {sorted(seeds)}")
            else:
                print(f"  {game}: {len(files)} files")

    print("=" * 50)


def load_and_process_data(agent_dir, data_column):
    """Load all CSV files and process them, maintaining game-seed structure."""
    game_files = organize_files_by_game(agent_dir)
    all_data = []

    for game, files in game_files.items():
        for filename, seed in files:
            try:
                filepath = os.path.join(agent_dir, filename)
                data = pd.read_csv(filepath)
                data['frame'] = pd.to_numeric(data['frame'], errors='coerce').fillna(0).astype(int)
                data[data_column] = pd.to_numeric(data[data_column], errors='coerce').fillna(0) * 100
                all_data.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    return all_data


def calculate_values(data_list, data_column, method='median', window_size=10):
    """Calculate median/mean values and apply moving average smoothing."""
    frames = sorted(set(frame for data in data_list for frame in data['frame'].values))
    values = []

    for frame in frames:
        frame_scores = []
        for data in data_list:
            if frame in data['frame'].values:
                frame_scores.append(data[data['frame'] == frame][data_column].values[0])

        if frame_scores:
            if method == 'median':
                values.append(np.median(frame_scores))
            elif method == 'mean':
                values.append(np.mean(frame_scores))
        else:
            values.append(np.nan)

    # Apply moving average smoothing
    series = pd.Series(values, index=frames)
    smoothed_series = series.rolling(window=window_size, min_periods=1).mean()

    return smoothed_series.index.tolist(), smoothed_series.values.tolist()


def percentage_fmt(x, pos):
    return f'{int(x)}%'  # Changed from \% to %


def plot_data(agent_data, data_column, ax):
    for agent_name, (frames, values) in agent_data.items():
        ax.plot(frames, values, label=agent_name, color=agent_colors[agent_name], linewidth=2)

    ax.set_xlabel('Millions of frames', fontsize=14)
    ax.set_ylabel(data_column.replace('_', ' ').capitalize(), fontsize=14)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(percentage_fmt))

    ax.set_xlim(left=0, right=200000000)
    ax.set_ylim(bottom=0)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    # Format x-axis to show millions
    def millions_formatter(x, pos):
        return f'{int(x / 1e6)}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_alpha(0.3)
        spine.set_linestyle('--')
        spine.set_linewidth(0.5)


check_files_coverage(agents_to_include, agent_dirs)

plotting_column = 'normalized_return'
agent_data_median = {}
agent_data_mean = {}

print("\nProcessing data for plotting...")

for agent in agents_to_include:
    agent_dir = agent_dirs[agent]
    if not os.path.exists(agent_dir):
        print(f"Skipping {agent}: Directory does not exist")
        continue

    print(f"Processing {agent}...")

    data_list = load_and_process_data(agent_dir, plotting_column)

    if data_list:
        # Calculate with smoothing (window size 10 as per README)
        frames_median, median_values = calculate_values(data_list, plotting_column, method='median', window_size=10)
        frames_mean, mean_values = calculate_values(data_list, plotting_column, method='mean', window_size=10)

        agent_data_median[agent] = (frames_median, median_values)
        agent_data_mean[agent] = (frames_mean, mean_values)

        print(f"  Processed {len(data_list)} game-seed combinations")
    else:
        print(f"  No valid data found for {agent}")

print("\nData processing complete. Creating plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plot_data(agent_data_median, 'Median human normalized return', axes[0])
plot_data(agent_data_mean, 'Mean human normalized return', axes[1])

handles, labels = axes[0].get_legend_handles_labels()

chronological_order = [
    'Double DQN',
    'Prioritized',
    'C51',
    'QRDQN',
    'Rainbow',
    'MDQN',
    'MIQN*',
    'Octopus'
]

ordered_handles = []
ordered_labels = []

for method in chronological_order:
    if method in labels:
        idx = labels.index(method)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

for i, label in enumerate(labels):
    if label not in chronological_order:
        ordered_handles.append(handles[i])
        ordered_labels.append(label)

fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=8, frameon=False,
           fontsize=13, columnspacing=1.0, handletextpad=0.5)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.subplots_adjust(bottom=0.22)

plt.savefig("normalized_score.pdf", format='pdf', bbox_inches='tight')
plt.show()

print("Plot saved successfully!")