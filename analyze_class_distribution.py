#!/usr/bin/env python
"""
统计NuScenes数据集中11个细类的点数分布
用于确定Open-set场景下的Unknown覆盖率策略
"""

import os
import os.path as osp
import pickle
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
import sys
sys.path.append('/home/Hash-Lee/paper3/3D_Openset_UDA')

from xmuda.data.nuscenes.nuscenes_dataloader import NuScenesBase


def analyze_class_distribution(preprocess_dir, splits, domain_name):
    """
    分析指定splits的类别分布
    
    Args:
        preprocess_dir: 预处理数据目录
        splits: 数据集划分，如 ('train_usa',) 或 ('train_singapore',)
        domain_name: 域名称，用于输出显示
    
    Returns:
        dict: 包含类别统计信息的字典
    """
    print(f"\n=== 分析 {domain_name} 域的类别分布 ===")
    print(f"数据划分: {splits}")
    
    # 初始化数据集
    dataset = NuScenesBase(
        split=splits,
        preprocess_dir=preprocess_dir,
        merge_classes=False  # 使用11个细类，不合并
    )
    
    # 获取类别名称
    class_names = dataset.class_names
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 统计每个类别的点数
    points_per_class = np.zeros(num_classes, dtype=np.int64)
    total_points = 0
    total_scenes = len(dataset.data)
    
    print(f"\n开始统计 {total_scenes} 个场景...")
    
    for i, data_dict in enumerate(dataset.data):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"处理进度: {i+1}/{total_scenes}")
        
        seg_labels = data_dict['seg_labels']
        
        # 统计当前场景的类别分布
        scene_class_counts = np.bincount(seg_labels, minlength=num_classes)
        points_per_class += scene_class_counts
        total_points += len(seg_labels)
    
    # 计算统计信息
    class_percentages = (points_per_class / total_points) * 100
    
    # 计算类别权重（log smoothed）
    class_weights = np.log(5 * points_per_class.sum() / (points_per_class + 1e-8))
    normalized_weights = class_weights / class_weights.min()
    
    # 构建结果字典
    results = {
        'domain_name': domain_name,
        'splits': splits,
        'class_names': class_names,
        'points_per_class': points_per_class,
        'total_points': total_points,
        'class_percentages': class_percentages,
        'class_weights': normalized_weights,
        'total_scenes': total_scenes
    }
    
    return results


def print_statistics(results):
    """打印详细的统计信息"""
    print(f"\n{'='*60}")
    print(f"域: {results['domain_name']}")
    print(f"总场景数: {results['total_scenes']}")
    print(f"总点数: {results['total_points']:,}")
    print(f"{'='*60}")
    
    print(f"\n{'类别':<20} {'点数':<15} {'占比(%)':<10} {'权重':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(results['class_names']):
        points = results['points_per_class'][i]
        percentage = results['class_percentages'][i]
        weight = results['class_weights'][i]
        print(f"{class_name:<20} {points:<15,} {percentage:<10.2f} {weight:<10.2f}")
    
    # 按点数排序显示
    print(f"\n按点数排序（从多到少）:")
    sorted_indices = np.argsort(results['points_per_class'])[::-1]
    for idx in sorted_indices:
        class_name = results['class_names'][idx]
        points = results['points_per_class'][idx]
        percentage = results['class_percentages'][idx]
        print(f"  {class_name:<20}: {points:>10,} 点 ({percentage:>6.2f}%)")


def suggest_unknown_coverage(source_results, target_results=None):
    """
    根据类别分布建议Unknown覆盖率策略
    """
    print(f"\n{'='*60}")
    print("Unknown覆盖率建议分析")
    print(f"{'='*60}")
    
    source_percentages = source_results['class_percentages']
    
    # 分析类别分布特征
    print(f"\n源域类别分布特征:")
    print(f"  最大类占比: {source_percentages.max():.2f}% ({source_results['class_names'][source_percentages.argmax()]})")
    print(f"  最小类占比: {source_percentages.min():.2f}% ({source_results['class_names'][source_percentages.argmin()]})")
    print(f"  平均类占比: {source_percentages.mean():.2f}%")
    print(f"  类别分布方差: {source_percentages.var():.2f}")
    
    # 建议不同的Unknown覆盖率
    print(f"\nUnknown覆盖率建议:")
    
    # 基于最小类的倍数
    min_percentage = source_percentages.min()
    print(f"  保守策略 (1x最小类): {min_percentage:.2f}%")
    print(f"  中等策略 (2x最小类): {min_percentage*2:.2f}%") 
    print(f"  激进策略 (3x最小类): {min_percentage*3:.2f}%")
    
    # 基于平均类的比例
    avg_percentage = source_percentages.mean()
    print(f"  基于平均 (0.5x平均): {avg_percentage*0.5:.2f}%")
    print(f"  基于平均 (1x平均): {avg_percentage:.2f}%")
    
    # 建议具体的类别作为Unknown候选
    sorted_indices = np.argsort(source_percentages)
    print(f"\n建议作为Unknown的候选类别（按点数从少到多）:")
    for i, idx in enumerate(sorted_indices[:5]):  # 显示最少的5个类
        class_name = source_results['class_names'][idx]
        percentage = source_percentages[idx]
        print(f"  {i+1}. {class_name}: {percentage:.2f}%")


def visualize_distribution(source_results, target_results=None, save_path=None):
    """
    可视化类别分布
    """
    plt.style.use('seaborn-v0_8')
    
    if target_results is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 源域分布
        ax1.bar(range(len(source_results['class_names'])), 
                source_results['class_percentages'],
                color='skyblue', alpha=0.8)
        ax1.set_title(f'{source_results["domain_name"]} 类别分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('占比 (%)')
        ax1.set_xticks(range(len(source_results['class_names'])))
        ax1.set_xticklabels(source_results['class_names'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 目标域分布
        ax2.bar(range(len(target_results['class_names'])), 
                target_results['class_percentages'],
                color='lightcoral', alpha=0.8)
        ax2.set_title(f'{target_results["domain_name"]} 类别分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('类别')
        ax2.set_ylabel('占比 (%)')
        ax2.set_xticks(range(len(target_results['class_names'])))
        ax2.set_xticklabels(target_results['class_names'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # 仅源域分布
        bars = ax1.bar(range(len(source_results['class_names'])), 
                       source_results['class_percentages'],
                       color='skyblue', alpha=0.8)
        ax1.set_title(f'{source_results["domain_name"]} 类别分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('占比 (%)')
        ax1.set_xticks(range(len(source_results['class_names'])))
        ax1.set_xticklabels(source_results['class_names'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, percentage in zip(bars, source_results['class_percentages']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='分析NuScenes类别分布')
    parser.add_argument('--preprocess_dir', 
                       default='/home/Hash-Lee/paper3/xMUDA/data/preprocess',
                       help='预处理数据目录')
    parser.add_argument('--output_dir',
                       default='/home/Hash-Lee/paper3/3D_Openset_UDA/analysis',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图表')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析源域 (USA)
    print("开始分析源域数据...")
    source_results = analyze_class_distribution(
        preprocess_dir=args.preprocess_dir,
        splits=('train_usa',),
        domain_name='Source (USA)'
    )
    
    print_statistics(source_results)
    
    # 尝试分析目标域 (Singapore) - 如果有GT的话
    target_results = None
    try:
        print("\n开始分析目标域数据...")
        target_results = analyze_class_distribution(
            preprocess_dir=args.preprocess_dir,
            splits=('train_singapore',),
            domain_name='Target (Singapore)'
        )
        print_statistics(target_results)
    except Exception as e:
        print(f"\n目标域数据分析失败（可能没有GT标签）: {e}")
        print("仅使用源域数据进行分析...")
    
    # 建议Unknown覆盖率策略
    suggest_unknown_coverage(source_results, target_results)
    
    # 保存详细统计结果
    results_file = osp.join(args.output_dir, 'class_distribution_analysis.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'source': source_results,
            'target': target_results
        }, f)
    print(f"\n详细统计结果已保存到: {results_file}")
    
    # 生成文本报告
    report_file = osp.join(args.output_dir, 'class_distribution_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("NuScenes 类别分布分析报告\n")
        f.write("="*50 + "\n\n")
        
        # 源域统计
        f.write(f"源域 ({source_results['domain_name']}):\n")
        f.write(f"总场景数: {source_results['total_scenes']}\n")
        f.write(f"总点数: {source_results['total_points']:,}\n\n")
        
        f.write("类别详细统计:\n")
        for i, class_name in enumerate(source_results['class_names']):
            points = source_results['points_per_class'][i]
            percentage = source_results['class_percentages'][i]
            weight = source_results['class_weights'][i]
            f.write(f"  {class_name}: {points:,} 点 ({percentage:.2f}%, 权重: {weight:.2f})\n")
        
        if target_results:
            f.write(f"\n目标域 ({target_results['domain_name']}):\n")
            f.write(f"总场景数: {target_results['total_scenes']}\n")
            f.write(f"总点数: {target_results['total_points']:,}\n\n")
            
            f.write("类别详细统计:\n")
            for i, class_name in enumerate(target_results['class_names']):
                points = target_results['points_per_class'][i]
                percentage = target_results['class_percentages'][i]
                f.write(f"  {class_name}: {points:,} 点 ({percentage:.2f}%)\n")
    
    print(f"文本报告已保存到: {report_file}")
    
    # 可视化
    if args.visualize:
        try:
            plot_file = osp.join(args.output_dir, 'class_distribution.png')
            visualize_distribution(source_results, target_results, save_path=plot_file)
        except Exception as e:
            print(f"可视化生成失败: {e}")
    
    return source_results, target_results


if __name__ == '__main__':
    main()
