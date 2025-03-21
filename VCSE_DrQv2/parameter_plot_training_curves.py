import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_and_concatenate_csvs(base_dir, algorithms):
    all_data = {alg: [] for alg in algorithms}
    for alg in algorithms:
        for i in range(1, 5):  # Assuming 4 runs for each algorithm
            run_dir = os.path.join(base_dir, f'do_rv-vcse=True_beta=0.1_seed={i}_{alg}')
            log_file = os.path.join(run_dir, 'eval.csv')

            if os.path.exists(log_file):
                data = pd.read_csv(log_file)

                # Check if 'frames' and 'return_mean' columns exist and are numeric
                if 'frame' in data.columns and 'episode_reward' in data.columns:
                    data = data[['frame', 'episode_reward']].dropna()
                    data['frame'] = pd.to_numeric(data['frame'], errors='coerce')
                    data['episode_reward'] = pd.to_numeric(data['episode_reward'], errors='coerce')
                    data = data.dropna()
                    all_data[alg].append(data)
                else:
                    print(f"Warning: Missing 'frame' or 'episode_reward' in {log_file}")

    for alg in algorithms:
        if all_data[alg]:
            all_data[alg] = pd.concat(all_data[alg])

    return all_data

def interpolate_data(data, x_col, y_col, num_points=500):
    x = data[x_col]
    y = data[y_col]

    interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')

    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = interp_func(x_new)

    return x_new, y_new


def plot_training_curves(base_dir, env_name, success_threshold):
    algorithms = ['alpha=0.1', 'alpha=0.3', 'alpha=0.5', 'alpha=0.7', 'alpha=0.9']
    colors = ['black', 'blue', 'red', 'green', 'purple']
    labels = ['α=0.1', 'α=0.3', 'α=0.5', 'α=0.7', 'α=0.9']

    all_data = read_and_concatenate_csvs(base_dir, algorithms)

    plt.figure(figsize=(8, 6))

    for alg, color, label in zip(algorithms, colors, labels):
        if len(all_data[alg]) > 0:
            # combined_data_grouped = all_data[alg].groupby('frame').mean().reset_index()
            # x_new, y_new = interpolate_data(combined_data_grouped, 'frame', 'episode_reward')
            # plt.plot(x_new, y_new, color=color, label=label)

            # Group by frame and calculate mean and std
            combined_data_grouped = all_data[alg].groupby('frame').agg(
                mean=('episode_reward', 'mean'),
                std=('episode_reward', 'std')
            ).reset_index()

            # Interpolate mean and std
            x_new, mean_new = interpolate_data(combined_data_grouped, 'frame', 'mean')
            _, std_new = interpolate_data(combined_data_grouped, 'frame', 'std')

            # Plot the mean curve
            plt.plot(x_new, mean_new, color=color, label=label)

            # Plot the shaded area for mean ± std
            plt.fill_between(x_new, mean_new - std_new, mean_new + std_new, color=color, alpha=0.2)

    plt.title(env_name)
    plt.xlabel('Environment Step')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{env_name}_parameter_analysis.png')
    plt.show()


base_dir = 'exp_fixed/cheetah_run-parameter_analysis'
env_name = 'Cheetah Run'
success_threshold = 0.8  # This threshold should be set based on the task's success criteria

# Plot the training curves
plot_training_curves(base_dir, env_name, success_threshold)
