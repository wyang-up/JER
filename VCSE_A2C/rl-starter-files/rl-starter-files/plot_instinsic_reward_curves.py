import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_and_concatenate_csvs(base_dir, algorithms):
    all_data = {alg: [] for alg in algorithms}
    for alg in algorithms:
        for i in range(1, 5):  # Assuming 4 runs for each algorithm
            run_dir = os.path.join(base_dir, f'MiniGrid-MultiRoom-N4-S5-v0-{alg}-{i}')
            log_file = os.path.join(run_dir, 'log.csv')

            if os.path.exists(log_file):
                data = pd.read_csv(log_file)

                # Check if 'frames' and 'return_mean' columns exist and are numeric
                if 'frames' in data.columns and 'entropy' in data.columns:
                    data = data[['frames', 'entropy']].dropna()
                    data['frames'] = pd.to_numeric(data['frames'], errors='coerce')
                    data['entropy'] = pd.to_numeric(data['entropy'], errors='coerce')
                    data = data.dropna()
                    all_data[alg].append(data)
                else:
                    print(f"Warning: Missing 'frames' or 'entropy' in {log_file}")

    for alg in algorithms:
        if all_data[alg]:
            all_data[alg] = pd.concat(all_data[alg])

    return all_data


# def compute_success_rate(data, threshold):
#     success = data['return_mean'] >= threshold
#     success_rate = success.mean() * 100  # Convert to percentage
#     return success_rate


def interpolate_data(data, x_col, y_col, num_points=500):
    x = data[x_col]
    y = data[y_col]

    interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')

    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = interp_func(x_new)

    return x_new, y_new


def plot_training_curves(base_dir, env_name, success_threshold):
    algorithms = ['original', 'sent', 'vcse','vae-vcse']
    colors = ['black', 'magenta', 'blue','red']
    labels = ['A2C', 'A2C+SE', 'A2C+VCSE','Our']

    all_data = read_and_concatenate_csvs(base_dir, algorithms)

    plt.figure(figsize=(8, 6))

    for alg, color, label in zip(algorithms, colors, labels):
        if len(all_data[alg]) > 0:
            combined_data_grouped = all_data[alg].groupby('frames').mean().reset_index()
            # combined_data_grouped['success_rate'] = combined_data_grouped.apply(
            #     lambda row: compute_success_rate(combined_data_grouped, success_threshold), axis=1)

            # Interpolate data for more even x-axis distribution
            x_new, y_new = interpolate_data(combined_data_grouped, 'frames', 'entropy')

            plt.plot(x_new, y_new, color=color, label=label)

            # Group by frame and calculate mean and std
            # combined_data_grouped = all_data[alg].groupby('frames').agg(
            #     mean=('entropy', 'mean'),
            #     std=('entropy', 'std')
            # ).reset_index()
            #
            # # Interpolate mean and std
            # x_new, mean_new = interpolate_data(combined_data_grouped, 'frames', 'mean')
            # _, std_new = interpolate_data(combined_data_grouped, 'frames', 'std')
            #
            # # Plot the mean curve
            # plt.plot(x_new, mean_new, color=color, label=label)
            #
            # # Plot the shaded area for mean ± std
            # plt.fill_between(x_new, mean_new - std_new, mean_new + std_new, color=color, alpha=0.2)

    plt.title(env_name)
    plt.xlabel('Environment Step')
    # plt.ylabel('Success Rate (%)')
    plt.ylabel('Entropy')
    # plt.ylim(0, 1)  # 设置y轴范围为0到1
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{env_name}_entropy_curves.png')
    plt.show()


base_dir = 'storage/MultiRoom-N4-S5'
env_name = 'MultiRoom-N4-S5'
success_threshold = 0.8  # This threshold should be set based on the task's success criteria

# Plot the training curves
plot_training_curves(base_dir, env_name, success_threshold)
