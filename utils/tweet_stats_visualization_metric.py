import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt


def visualize_tweet_stats(data_file: str, save_path: str, plot_now: bool = False):
    """
    Visualize the collected stats (likes, reposts, comments) over timesteps.

    Args:
        data_file (str): Path to the .npy file containing the stats data.
        save_path (str): Path to save the plot as an image file.
        plot_now (bool): Whether to show the plot immediately.
    """
    colors = ['blue', 'green', 'red', 'orange']  # Match the colors used in the plot lines
    # Load the saved stats data from the .npy file
    stats_data = np.load(data_file)

    # Assuming the number of timesteps is the same as the number of rows in stats_data
    num_timesteps = stats_data.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot likes, reposts, and comments over timesteps
    ax.plot(range(1, num_timesteps + 1),
            stats_data[:, 0], label='Likes', color=colors[0], linestyle='--')
    ax.plot(range(1, num_timesteps + 1),
            stats_data[:, 1], label='Reposts', color=colors[1], linestyle='-.')
    ax.plot(range(1, num_timesteps + 1),
            stats_data[:, 2], label='Good Comments', color=colors[2], linestyle=':')
    ax.plot(range(1, num_timesteps + 1),
            stats_data[:, 3], label='Bad Comments', color=colors[3], linestyle='-')
    # ax.plot(range(1, num_timesteps + 1),
    #         stats_data[:, 4], label='Flags', color='purple', linestyle='-')
    # ax.plot(range(1, num_timesteps + 1),
    #         stats_data[:, 5], label='Views', color='brown', linestyle='-')

    # Customize plot
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Count')
    ax.set_title('Social Media Post Stats Over Time')
    ax.legend()

    # Add the vertical axis value annotation for the last time point with matching colors
    for i, label in enumerate(['Likes', 'Reposts', 'Good Comments', 'Bad Comments']):
        ax.annotate(f'{int(stats_data[-1, i])}', 
                   xy=(num_timesteps, stats_data[-1, i]),
                   xytext=(5, 0), 
                   textcoords='offset points',
                   ha='left',
                   va='center',
                   color=colors[i])  # Use the same color as the corresponding line

    # If a save path is provided, save the figure
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")

    # Show the plot
    if plot_now:
        plt.show()


def visualize_multiple_tweet_stats(data_paths: Union[Dict, str], 
                                   save_path: str = None, 
                                   filter_str: str = None,
                                   title: str = 'Average Social Media Post Stats Over Time with 95% CI',
                                   show_likes: bool = True,
                                   show_reposts: bool = True,
                                   show_good_comments: bool = True,
                                   show_bad_comments: bool = True,
                                   plot_now: bool = True,
                                   dataset_label: str = None):
    """
    Visualize the average stats (likes, reposts, comments) with 95% confidence intervals over timesteps
    for multiple experiments.

    Args:
        data_paths (list or str): List of paths to .npy files or a folder containing .npy files.
        save_path (str): Path to save the plot as an image file.
        filter_str (str): String to filter files by.
        title (str): Title of the plot.
        show_likes (bool): Whether to show likes data.
        show_reposts (bool): Whether to show reposts data.
        show_good_comments (bool): Whether to show good comments data.
        show_bad_comments (bool): Whether to show bad comments data.
        plot_now (bool): Whether to show the plot.
        dataset_label (str): Optional label for the dataset to prepend to the metric names.
    """
    # If data_paths is a folder, get all .npy files in the folder
    if isinstance(data_paths, str) and os.path.isdir(data_paths):
        data_paths = [os.path.join(data_paths, f) for f in os.listdir(
            data_paths) if f.endswith('.npy')]
    print(data_paths)
    if filter_str:
        data_paths = [f for f in data_paths if filter_str in f]
    # List to store the stats data from all experiments
    all_stats = []
    print(data_paths)
    # Load the data from each file
    for file in data_paths:
        stats_data = np.load(file)
        all_stats.append(stats_data)

    # Convert list to numpy array (shape: [num_experiments, num_timesteps, 4])
    all_stats = np.array(all_stats)

    # Calculate the average and 95% CI for each timestep
    num_timesteps = all_stats.shape[1]
    avg_stats = np.mean(all_stats, axis=0)
    std_stats = np.std(all_stats, axis=0)

    # Calculate 95% CI (using t-distribution for CI)
    confidence_interval = stats.t.interval(
        0.90, all_stats.shape[0] - 1, loc=avg_stats, scale=std_stats/np.sqrt(all_stats.shape[0]))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot likes, reposts, and comments with average and CI
    if show_likes:
        label = f"{dataset_label} - Likes" if dataset_label else "Likes (Avg)"
        ax.plot(range(1, num_timesteps + 1),
                avg_stats[:, 0], label=label, color='blue', linestyle='--')
        # Add value annotation for the last point
        ax.annotate(f'{avg_stats[-1, 0]:.1f}', 
                   xy=(num_timesteps, avg_stats[-1, 0]), 
                   xytext=(5, 0), 
                   textcoords='offset points',
                   ha='left', va='center')
    
    if show_reposts:
        label = f"{dataset_label} - Reposts" if dataset_label else "Reposts (Avg)"
        ax.plot(range(1, num_timesteps + 1),
                avg_stats[:, 1], label=label, color='green', linestyle='-.')
        # Add value annotation for the last point
        ax.annotate(f'{avg_stats[-1, 1]:.1f}', 
                   xy=(num_timesteps, avg_stats[-1, 1]), 
                   xytext=(5, 0), 
                   textcoords='offset points',
                   ha='left', va='center')
                
    if show_good_comments:
        label = f"{dataset_label} - Good Comments" if dataset_label else "Good Comments (Avg)"
        ax.plot(range(1, num_timesteps + 1),
                avg_stats[:, 2], label=label, color='red', linestyle=':')
        # Add value annotation for the last point
        ax.annotate(f'{avg_stats[-1, 2]:.1f}', 
                   xy=(num_timesteps, avg_stats[-1, 2]), 
                   xytext=(5, 0), 
                   textcoords='offset points',
                   ha='left', va='center')
                
    if show_bad_comments:
        label = f"{dataset_label} - Bad Comments" if dataset_label else "Bad Comments (Avg)"
        ax.plot(range(1, num_timesteps + 1),
                avg_stats[:, 3], label=label, color='orange', linestyle='-')
        # Add value annotation for the last point
        ax.annotate(f'{avg_stats[-1, 3]:.1f}', 
                   xy=(num_timesteps, avg_stats[-1, 3]), 
                   xytext=(5, 0), 
                   textcoords='offset points',
                   ha='left', va='center')

    # Add the confidence interval as shaded areas
    if show_likes:
        ax.fill_between(range(1, num_timesteps + 1),
                        confidence_interval[0][:, 0], confidence_interval[1][:, 0], color='blue', alpha=0.2)
    if show_reposts:
        ax.fill_between(range(1, num_timesteps + 1),
                        confidence_interval[0][:, 1], confidence_interval[1][:, 1], color='green', alpha=0.2)
    if show_good_comments:
        ax.fill_between(range(1, num_timesteps + 1),
                        confidence_interval[0][:, 2], confidence_interval[1][:, 2], color='red', alpha=0.2)
    if show_bad_comments:
        ax.fill_between(range(1, num_timesteps + 1),
                        confidence_interval[0][:, 3], confidence_interval[1][:, 3], color='orange', alpha=0.2)

    # Customize plot
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    # If a save path is provided, save the figure
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")

    # Show the plot
    if plot_now:
        plt.show()
        
    return fig, ax
        
def visualize_comparison(data_parent_dir: str, save_path: str = None, 
                        title: str = 'Comparison of Different Configurations',
                        show_likes: bool = True, show_reposts: bool = True,
                        show_good_comments: bool = True, show_bad_comments: bool = True,
                        plot_now: bool = True):
    """
    Create a visualization comparing single, centralized, and decentralized configurations.
    
    Args:
        data_parent_dir (str): Parent directory containing the data files.
        save_path (str): Path to save the figure.
        title (str): Title of the plot.
        show_likes (bool): Whether to show likes data.
        show_reposts (bool): Whether to show reposts data.
        show_good_comments (bool): Whether to show good comments data.
        show_bad_comments (bool): Whether to show bad comments data.
        plot_now (bool): Whether to show the plot immediately.
    """
    # Define filter strings for each configuration
    no_defense_filter = 'post_stats_data_test_1000_good_bad_random_bernoulli_wlx_nodefense'
    prebunking_filter = 'post_stats_data_test_1000_good_bad_random_bernoulli_wlx_prebunking'
    debunking_filter = 'post_stats_data_test_1000_good_bad_random_bernoulli_wlx_debunking'
    ban_filter = 'post_stats_data_test_1000_good_bad_random_bernoulli_wlx_ban_nomessage'

    # Define colors for each configuration
    colors = {
        'Without Defense': ['blue', 'lightblue'],
        'Pre-bunking': ['green', 'lightgreen'],
        'De-bunking': ['red', 'lightcoral'],
        'Banning': ['purple', 'lightpurple']
    }
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Add data for each configuration
    configs = [
        ('Without Defense', no_defense_filter, colors['Without Defense']),
        ('Pre-bunking', prebunking_filter, colors['Pre-bunking']),
        ('De-bunking', debunking_filter, colors['De-bunking']),
        ('Banning', ban_filter, colors['Banning'])
    ]
    
    for config_name, filter_str, color_pair in configs:
        # Load data for this configuration
        data_paths = [os.path.join(data_parent_dir, f) for f in os.listdir(data_parent_dir) if f.endswith('.npy')]
        data_paths = [f for f in data_paths if filter_str in f]
        
        # Load the data from each file
        all_stats = []
        for file in data_paths:
            stats_data = np.load(file)
            all_stats.append(stats_data)
        
        # Convert list to numpy array
        all_stats = np.array(all_stats)
        
        # Calculate the average and 95% CI for each timestep
        num_timesteps = all_stats.shape[1]
        avg_stats = np.mean(all_stats, axis=0)
        std_stats = np.std(all_stats, axis=0)
        
        # Calculate 95% CI
        confidence_interval = stats.t.interval(
            0.90, all_stats.shape[0] - 1, loc=avg_stats, scale=std_stats/np.sqrt(all_stats.shape[0]))
        
        # Plot the data for this configuration
        stat_idx = []
        if show_likes: stat_idx.append((0, f"{config_name} - Likes", '--'))
        if show_reposts: stat_idx.append((1, f"{config_name} - Reposts", '-.'))
        if show_good_comments: stat_idx.append((2, f"{config_name} - Good Comments", ':'))
        if show_bad_comments: stat_idx.append((3, f"{config_name} - Bad Comments", '-'))
        
        for idx, label, linestyle in stat_idx:
            ax.plot(range(1, num_timesteps + 1), avg_stats[:, idx], 
                    label=label, color=color_pair[0], linestyle=linestyle)
            # Add value annotation for the last point
            # ax.annotate(f'{avg_stats[-1, idx]:.1f}', 
            #            xy=(num_timesteps, avg_stats[-1, idx]), 
            #            xytext=(5, 0), 
            #            textcoords='offset points',
            #            ha='left', va='center',
            #            color=color_pair[0])
            ax.fill_between(range(1, num_timesteps + 1),
                            confidence_interval[0][:, idx], confidence_interval[1][:, idx], 
                            color=color_pair[0], alpha=0.1)
    
    # Customize plot
    ax.set_xlabel('Timestep', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    # ax.set_title(title, fontsize=20)
    ax.legend(fontsize=20)
    # ax.legend()
    ax.tick_params(axis='x', labelsize=20)  # x轴刻度数字大小
    ax.tick_params(axis='y', labelsize=20)
    
    
    # If a save path is provided, save the figure
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")
    
    # Show the plot
    if plot_now:
        plt.show()
        
    return fig, ax

def save_tweet_stats_visualization(save_paths: List[str], data_parent_dir: str):
    """
    Save the tweet stats visualization to the given save paths.
    """
    for save_path in save_paths:
        if "/both" in save_path:
            # New comparison visualization showing all configurations
            title = 'Comparison of Different Defense Settings'
            visualize_comparison(
                data_parent_dir=data_parent_dir,
                save_path=save_path,
                title=title,
                show_likes="likes" in save_path,
                show_reposts="reposts" in save_path,
                show_good_comments="good_comments" in save_path,
                show_bad_comments="bad_comments" in save_path,
                plot_now=False
            )
            continue
        elif "/centralized" in save_path:
            filter_str = 'post_stats_data_test_1000_good_bad_member_random_bernoulli_wlx'
            title='Centralized Group'
        elif "/decentralized" in save_path:
            filter_str = 'post_stats_data_test_1000_good_bad_random_bernoulli_wlx'
            title='Decentralized Group'
        elif "/single" in save_path:
            filter_str = 'post_stats_data_test_901_good_alone_bad_random_bernoulli_wlx'
            title='Single'
        elif "/only_good" in save_path:
            filter_str = 'post_stats_data_test_901_good_onlygood_bad_random_bernoulli_wlx'
            title='Only Good'
        visualize_multiple_tweet_stats(
            data_paths=data_parent_dir,
            filter_str=filter_str,
            save_path=save_path,
            title=title,
            show_likes="likes" in save_path,
            show_reposts="reposts" in save_path,
            show_good_comments="good_comments" in save_path,
            show_bad_comments="bad_comments" in save_path,
            plot_now=False
        )



def compare_visualize_tweet_stats(data_dict: Dict[str, str],
                            save_path=None,
                            title='Comparison of Social Media Post Stats Over Time',
                            show_likes=True,
                            show_reposts=True,
                            show_good_comments=True,
                            show_bad_comments=True):
    """
    对比可视化多组实验的统计数据 (likes, reposts, good comments, bad comments) 随时间步的变化。

    Args:
        data_dict (dict): 字典，键为标签（label），值为对应的 .npy 数据文件路径。
        save_path (str, optional): 保存图形的文件路径。默认为 None，不保存。
        title (str, optional): 图表标题。
    """
    # 检查字典是否为空
    if not data_dict:
        raise ValueError("data_dict 不能为空。")

    # 初始化绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 定义线条样式和颜色
    line_styles = ['--', '-.', ':', '-']
    colors = [
        'blue',        # 蓝色（冷色调，中等亮度）
        'orange',      # 橙色（暖色调，明亮）
        'green',       # 绿色（冷色调，中等亮度）
        'red',         # 红色（暖色调，鲜艳）
        'purple',      # 紫色（冷色调，深沉）
        'brown',       # 棕色（暖色调，暗沉）
        'pink',        # 粉色（暖色调，柔和）
        'gray',        # 灰色（中性色，中等亮度）
        'olive',       # 橄榄色（暖色调，暗绿）
        'cyan',        # 青色（冷色调，明亮）
        'navy',        # 海军蓝（冷色调，深沉）
        'skyblue',     # 天蓝色（冷色调，浅亮）
        'darkred',     # 深红色（暖色调，暗沉）
        'lightblue',   # 浅蓝（冷色调，柔和）
        'darkgreen',   # 深绿（冷色调，深沉）
        'lightgreen',  # 浅绿（冷色调，明亮）
        'yellow',      # 黄色（暖色调，鲜艳）
        'magenta',     # 品红（暖色调，鲜艳）
        'lime',        # 酸橙色（冷色调，明亮）
        'teal',        # 蓝绿色（冷色调，中等亮度）
        'gold',        # 金色（暖色调，明亮）
        'indigo',      # 靛蓝（冷色调，深沉）
        'salmon',      # 鲑鱼色（暖色调，柔和）
        'lavender',    # 薰衣草色（冷色调，浅淡）
        'black',       # 黑色（中性色，最暗）
    ]

    # 初始化样式和颜色索引
    style_index = 0
    color_index = 0

    # 遍历字典的键值对
    for label, data_file in data_dict.items():
        # 加载数据
        stats_data = np.load(data_file)
        num_timesteps = stats_data.shape[0]

        offset = 0

        # 绘制 Likes, Reposts, Good Comments, Bad Comments 4条曲线
        if show_likes:
            ax.plot(range(1, num_timesteps + 1), stats_data[:, 0], label=f'{label} - Likes',
                    linestyle=line_styles[style_index], color=colors[color_index])
            offset += 1
        if show_reposts:
            ax.plot(range(1, num_timesteps + 1), stats_data[:, 1], label=f'{label} - Reposts',
                    linestyle=line_styles[(style_index + offset) % len(line_styles)], color=colors[(color_index + offset) % len(colors)])
            offset += 1
        if show_good_comments:
            ax.plot(range(1, num_timesteps + 1), stats_data[:, 2], label=f'{label} - Good Comments',
                    linestyle=line_styles[(style_index + offset) % len(line_styles)], color=colors[(color_index + offset) % len(colors)])
            offset += 1
        if show_bad_comments:
            ax.plot(range(1, num_timesteps + 1), stats_data[:, 3], label=f'{label} - Bad Comments',
                    linestyle=line_styles[(style_index + offset) % len(line_styles)], color=colors[(color_index + offset) % len(colors)])
            offset += 1

        # 更新样式和颜色索引
        style_index = (style_index + offset) % len(line_styles)
        color_index = (color_index + offset) % len(colors)

    # 设置图表属性
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Count')
    ax.set_title(f'{title}')
    ax.legend()

    # 保存图形（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # data_dict = {
    #     "Single": "results\post_stats_data_test_901_good_alone_bad_random_bernoulli_xst_2025-02-27_03-15-02.npy",
    #     "Centralized Group": "results\post_stats_data_test_1000_good_bad_member_random_bernoulli_xst_2025-02-27_04-23-14.npy",
    #     "Decentralized Group": "results\post_stats_data_test_1000_good_bad_random_bernoulli_xst_2025-02-27_10-23-17.npy"
    # }
    # compare_visualize_tweet_stats(data_dict,
    #                         save_path=None,
    #                         title='',
    #                         show_likes=True,
    #                         show_reposts=False,
    #                         show_good_comments=False,
    #                         show_bad_comments=False)
    # visualize_tweet_stats("results/post_stats_data_test_120_single_bad_random_bernoulli_xst_2025-02-15_05-20-47.npy", None)
    result_path = r"experiment_results/metric"
    result_path_str = result_path.replace('/', '_')
    print(result_path_str)
    save_paths = [
        # f"results/misinformation/centralized_likes_{result_path_str}.png",
        # f"results/misinformation/decentralized_likes_{result_path_str}.png",
        # f"results/misinformation/single_likes_{result_path_str}.png",
        # f"results/misinformation/centralized_reposts_{result_path_str}.png",
        # f"results/misinformation/decentralized_reposts_{result_path_str}.png",
        # f"results/misinformation/single_reposts_{result_path_str}.png",
        # f"results/misinformation/centralized_good_comments_{result_path_str}.png",
        # f"results/misinformation/decentralized_good_comments_{result_path_str}.png",
        # f"results/misinformation/single_good_comments_{result_path_str}.png",
        # f"results/misinformation/centralized_bad_comments_{result_path_str}.png",
        # f"results/misinformation/decentralized_bad_comments_{result_path_str}.png",
        # f"results/misinformation/single_bad_comments_{result_path_str}.png",
        # f"results/misinformation/both_bad_comments_{result_path_str}.png",
        # f"results/misinformation/both_good_comments_{result_path_str}.png",
        # f"results/misinformation/both_reposts_{result_path_str}.png",
        # f"results/misinformation/both_likes_{result_path_str}.png",
        f"experiment_results/metric/both_reposts_{result_path_str}.png",
        f"experiment_results/metric/both_likes_{result_path_str}.png",
        f"experiment_results/metric/both_good_comments_{result_path_str}.png",
        f"experiment_results/metric/both_bad_comments_{result_path_str}.png"
    ]
    save_tweet_stats_visualization(save_paths=save_paths, data_parent_dir=result_path)
