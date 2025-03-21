import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_and_concatenate_csvs(base_dir, algorithms):
    all_data = {alg: [] for alg in algorithms}
    for alg in algorithms:
        for i in range(1, 5):  # Assuming 4 runs for each algorithm
            run_dir = os.path.join(base_dir, f'do_{alg}_beta=0.1_seed={i}')
            log_file = os.path.join(run_dir, 'train.csv')

            if os.path.exists(log_file):
                data = pd.read_csv(log_file)

                # Check if 'frames' and 'return_mean' columns exist and are numeric
                if 'frame' in data.columns and 'intrinsic_reward' in data.columns:
                    data = data[['frame', 'intrinsic_reward']].dropna()
                    data['frame'] = pd.to_numeric(data['frame'], errors='coerce')
                    data['intrinsic_reward'] = pd.to_numeric(data['intrinsic_reward'], errors='coerce')
                    data = data.dropna()
                    all_data[alg].append(data)
                else:
                    print(f"Warning: Missing 'frame' or 'intrinsic_reward' in {log_file}")

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
    algorithms = ['vcse=False', 'vcse=True', 'rv-vcse=True']
    colors = ['black', 'blue','red']
    labels = ['DrQv2+SE', 'DrQv2+VCSE','DrQv2+JER']

    all_data = read_and_concatenate_csvs(base_dir, algorithms)

    plt.figure(figsize=(8, 6))

    for alg, color, label in zip(algorithms, colors, labels):
        if len(all_data[alg]) > 0:
            combined_data_grouped = all_data[alg].groupby('frame').mean().reset_index()
            x_new, y_new = interpolate_data(combined_data_grouped, 'frame', 'intrinsic_reward')

            plt.plot(x_new, y_new, color=color, label=label)

    plt.title(env_name)
    plt.xlabel('Environment Step')
    plt.ylabel('Intrinsic  Reward')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{env_name}_intrinsic_reward.png')
    plt.show()


base_dir = 'exp_fixed/walker_run'
env_name = 'walker_run'
success_threshold = 0.8  # This threshold should be set based on the task's success criteria

# Plot the training curves
plot_training_curves(base_dir, env_name, success_threshold)
