import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_and_average_rewards(base_dir, algorithms):
    """
    遍历每个算法目录，计算所有随机种子 train.csv 中的 extrinsic_reward 总和，并取平均值。
    """
    results = []

    for alg in algorithms:
        alg_data = []

        # 遍历随机种子目录
        for i in range(1, 5):  # 假设有 4 个随机种子
            run_dir = os.path.join(base_dir, f'do_{alg}_beta=0.1_seed={i}')
            log_file = os.path.join(run_dir, 'train.csv')

            if os.path.exists(log_file):
                # 读取 train.csv 文件
                data = pd.read_csv(log_file)

                if 'extrinsic_reward' in data.columns:
                    total_reward = data['extrinsic_reward'].sum()
                    alg_data.append(total_reward)
                else:
                    print(f"Warning: Missing 'extrinsic_reward' in {log_file}")

        # 对每个算法的随机种子取平均值
        if len(alg_data) > 0:
            avg_reward = np.mean(alg_data)
            results.append({'algorithm': alg, 'avg_reward': avg_reward})

    return results


def plot_bar_chart(results, tasks, algorithms, colors, labels):
    """
    绘制柱状图：横轴为任务名称，纵轴为 extrinsic_reward 的平均总和，不同颜色代表不同算法。
    """
    x = np.arange(len(tasks))  # 横轴任务的位置
    bar_width = 0.2  # 每个柱子的宽度

    plt.figure(figsize=(10, 6))

    for i, alg in enumerate(algorithms):
        # 从结果中过滤出当前算法的数据
        alg_rewards = [result['avg_reward'] for result in results if result['algorithm'] == alg]
        plt.bar(x + i * bar_width, alg_rewards, bar_width, label=labels[i], color=colors[i])

    plt.xlabel("Tasks")
    plt.ylabel("Cumulative Extrinsic Reward")
    plt.title("Cumulative Extrinsic Reward per DMC Control Tasks and Algorithm")
    plt.xticks(x + bar_width * (len(algorithms) - 1) / 2, tasks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'dm_control_bar_chart.png')
    plt.show()


# 基础目录和算法配置
base_dir = 'exp_fixed'
tasks = ['Quadruped_Walk', 'Walker_Run', 'Cheetah_Run', 'Cartpole_Swingup_Sparse', 'Pendulum_Swingup', 'Reacher_Hard']  # 示例任务，可根据您的目录修改
algorithms = ['vcse=False', 'vcse=True', 'rv-vcse=True']  # 示例算法
colors = ['green', 'blue', 'red']
labels = ['DrQv2+SE', 'DrQv2+VCSE', 'DrQv2+AJER']

# 遍历所有任务，收集结果
all_results = []
for task in tasks:
    task_dir = os.path.join(base_dir, task)
    task_results = read_and_average_rewards(task_dir, algorithms)

    # 添加任务名称到结果中
    for result in task_results:
        result['task'] = task

    all_results.extend(task_results)

# 绘制柱状图
plot_bar_chart(all_results, tasks, algorithms, colors, labels)
