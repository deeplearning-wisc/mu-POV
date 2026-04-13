#!/usr/bin/env python3
"""
Dashboard utilities for visualizing results and accuracy trends.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


class Tee:
    """A class that writes to both file and console."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()


def display_results(args, all_results, exp_name):

    scores = np.array([result['scores'] for result in all_results])
    single_agent_scores_raw = np.array([result['single_agent_scores'] for result in all_results])

    one_dim_scores = False
    if len(scores.shape) == 1:
        one_dim_scores = True
        scores = np.expand_dims(scores, axis=-1)
        single_agent_scores_raw = np.expand_dims(single_agent_scores_raw, axis=-1)

    avg_scores = np.mean(scores, axis=0)

    # single_agent_scores shape: (num_agents, num_metrics)
    num_agents = single_agent_scores_raw.shape[1]
    multi_agent = num_agents > 1

    single_agent_scores = np.mean(single_agent_scores_raw, axis=0)
    single_mean = np.mean(single_agent_scores, axis=0)
    single_std = np.std(single_agent_scores, axis=0)

    # p-value only meaningful when there are multiple distinct agents
    p_values = None
    if multi_agent:
        p_vals = []
        for metric_idx in range(avg_scores.shape[0]):
            voted_series = scores[:, metric_idx]
            baseline_series = single_agent_scores_raw[:, :, metric_idx].mean(axis=1)
            _, p_val = ttest_rel(voted_series, baseline_series, nan_policy='omit')
            p_vals.append(p_val)
        p_vals = np.array(p_vals)
        if not np.all(np.isnan(p_vals)):
            p_values = p_vals

    num_samples = len(all_results)
    time_steps = np.arange(1, num_samples + 1)
    voted_cumulative = np.cumsum(scores, axis=0) / time_steps[:, np.newaxis]

    if multi_agent:
        agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    if args.data in args.closed_ended_tasks + args.math_reasoning_tasks:
        mv_scores = np.array([result['mv_scores'] for result in all_results], dtype=int)
        if one_dim_scores:
            mv_scores = np.expand_dims(mv_scores, axis=-1)

        if multi_agent:
            print(f"Accuracy: {avg_scores[0]*100:.2f}  ||  Single Agent: {single_mean[0]*100:.2f} ± {single_std[0]*100:.2f}  ||  Majority Vote: {mv_scores.mean()*100:.2f}")
        else:
            print(f"Accuracy: {avg_scores[0]*100:.2f}")
        if p_values is not None:
            print(f"p_value: Accuracy={p_values[0]:.4f}")
        print()

        for i in range(num_agents):
            print(f"Agent {i+1}: Accuracy={single_agent_scores[i][0]*100:.2f}")

        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes.plot(time_steps, voted_cumulative[:, 0],
                  color='#1f77b4', linewidth=3, label='Final Score', linestyle='-', marker='o', markersize=4)
        if multi_agent:
            agent_cumulative = np.zeros((num_samples, num_agents + 1, 1))
            for agent_idx in range(num_agents):
                agent_cumulative[:, agent_idx, :] = np.cumsum(single_agent_scores_raw[:, agent_idx, :], axis=0) / time_steps[:, np.newaxis]
            agent_cumulative[:, num_agents, :] = np.cumsum(mv_scores, axis=0) / time_steps[:, np.newaxis]
            for agent_idx in range(num_agents):
                axes.plot(time_steps, agent_cumulative[:, agent_idx, 0],
                          color=agent_colors[agent_idx], linewidth=2,
                          label=f'Agent {agent_idx + 1}', alpha=0.7, linestyle='--')
            axes.plot(time_steps, agent_cumulative[:, num_agents, 0],
                      color='magenta', linewidth=2, label='Majority Vote', alpha=0.7, linestyle='--')
        axes.set_xlabel('Sample Number', fontsize=11)
        axes.set_ylabel('Accuracy Score', fontsize=11)
        axes.set_title('Accuracy Trajectory', fontsize=12, fontweight='bold')
        axes.grid(True, alpha=0.3)
        axes.legend(fontsize=9)

    elif args.data in args.code_generation_tasks:
        if multi_agent:
            print(f"Pass@1: {avg_scores[0]*100:.2f}  ||  Single Agent: {single_mean[0]*100:.2f} ± {single_std[0]*100:.2f}")
            print(f"BLEU: {avg_scores[1]*100:.2f}  ||  Single Agent: {single_mean[1]*100:.2f} ± {single_std[1]*100:.2f}")
        else:
            print(f"Pass@1: {avg_scores[0]*100:.2f}")
            print(f"BLEU: {avg_scores[1]*100:.2f}")
        if p_values is not None:
            print(f"p_value: Pass@1={p_values[0]:.4f}  ||  BLEU={p_values[1]:.4f}")
        print()

        for i in range(num_agents):
            print(f"Agent {i+1}: Pass@1={single_agent_scores[i][0]*100:.2f}  ||  BLEU={single_agent_scores[i][1]*100:.2f}")

        metric_names = ['Pass@1', 'BLEU']
        colors = ['#1f77b4', '#ff7f0e']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if multi_agent:
            agent_cumulative = np.zeros((num_samples, num_agents, 2))
            for agent_idx in range(num_agents):
                agent_cumulative[:, agent_idx, :] = np.cumsum(single_agent_scores_raw[:, agent_idx, :], axis=0) / time_steps[:, np.newaxis]

        for metric_idx, (metric_name, color) in enumerate(zip(metric_names, colors)):
            axes[metric_idx].plot(time_steps, voted_cumulative[:, metric_idx],
                                  color=color, linewidth=3, label='Final Score', linestyle='-', marker='o', markersize=4)
            if multi_agent:
                for agent_idx in range(num_agents):
                    axes[metric_idx].plot(time_steps, agent_cumulative[:, agent_idx, metric_idx],
                                          color=agent_colors[agent_idx], linewidth=2,
                                          label=f'Agent {agent_idx + 1}', alpha=0.7, linestyle='--')
            axes[metric_idx].set_xlabel('Sample Number', fontsize=11)
            axes[metric_idx].set_ylabel(f'{metric_name} Score', fontsize=11)
            axes[metric_idx].set_title(f'{metric_name} Trajectory', fontsize=12, fontweight='bold')
            axes[metric_idx].grid(True, alpha=0.3)
            axes[metric_idx].legend(fontsize=9)

    elif args.data in args.text_generation_tasks:
        if multi_agent:
            print(f"ROUGE 1: {avg_scores[0]*100:.2f}  ||  Single Agent: {single_mean[0]*100:.2f} ± {single_std[0]*100:.2f}")
            print(f"ROUGE 2: {avg_scores[1]*100:.2f}  ||  Single Agent: {single_mean[1]*100:.2f} ± {single_std[1]*100:.2f}")
            print(f"ROUGE L: {avg_scores[2]*100:.2f}  ||  Single Agent: {single_mean[2]*100:.2f} ± {single_std[2]*100:.2f}")
            print(f"BLEU: {avg_scores[3]*100:.2f}  ||  Single Agent: {single_mean[3]*100:.2f} ± {single_std[3]*100:.2f}")
        else:
            print(f"ROUGE 1: {avg_scores[0]*100:.2f}")
            print(f"ROUGE 2: {avg_scores[1]*100:.2f}")
            print(f"ROUGE L: {avg_scores[2]*100:.2f}")
            print(f"BLEU: {avg_scores[3]*100:.2f}")
        if p_values is not None:
            print(f"p_value: R1={p_values[0]:.4f}  ||  R2={p_values[1]:.4f}  ||  RL={p_values[2]:.4f}  ||  BLEU={p_values[3]:.4f}")
        print()

        for i in range(num_agents):
            print(f"Agent {i+1}: R1={single_agent_scores[i][0]*100:.2f} || R2={single_agent_scores[i][1]*100:.2f} || RL={single_agent_scores[i][2]*100:.2f} || BLEU={single_agent_scores[i][3]*100:.2f}")

        metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        if multi_agent:
            agent_cumulative = np.zeros((num_samples, num_agents, 4))
            for agent_idx in range(num_agents):
                agent_cumulative[:, agent_idx, :] = np.cumsum(single_agent_scores_raw[:, agent_idx, :], axis=0) / time_steps[:, np.newaxis]

        for metric_idx, (metric_name, color) in enumerate(zip(metric_names, colors)):
            axes[metric_idx].plot(time_steps, voted_cumulative[:, metric_idx],
                                  color=color, linewidth=3, label='Final Score', linestyle='-', marker='o', markersize=4)
            if multi_agent:
                for agent_idx in range(num_agents):
                    axes[metric_idx].plot(time_steps, agent_cumulative[:, agent_idx, metric_idx],
                                          color=agent_colors[agent_idx], linewidth=2,
                                          label=f'Agent {agent_idx + 1}', alpha=0.7, linestyle='--')
            axes[metric_idx].set_xlabel('Sample Number', fontsize=11)
            axes[metric_idx].set_ylabel(f'{metric_name} Score', fontsize=11)
            axes[metric_idx].set_title(f'{metric_name} Trajectory', fontsize=12, fontweight='bold')
            axes[metric_idx].grid(True, alpha=0.3)
            axes[metric_idx].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'out/{exp_name}/score_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()

    return avg_scores


def plot_similarity_trajectories(args, similarity_trajectories, exp_name):

    num_samples = len(similarity_trajectories['min'])
    if num_samples == 0:
        return

    num_steps = len(similarity_trajectories['min'][0])
    num_agents = len(similarity_trajectories['min'][0][0]) if num_steps > 0 else 0

    if num_steps == 0 or num_agents == 0:
        return

    min_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['min']])
    bon_min_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['bon_min']])
    max_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['max']])
    bon_max_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['bon_max']])
    mean_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['mean']])
    bon_mean_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['bon_mean']])

    min_mean = np.mean(min_trajectories, axis=0)
    bon_min_mean = np.mean(bon_min_trajectories, axis=0)
    max_mean = np.mean(max_trajectories, axis=0)
    bon_max_mean = np.mean(bon_max_trajectories, axis=0)
    mean_mean = np.mean(mean_trajectories, axis=0)
    bon_mean_mean = np.mean(bon_mean_trajectories, axis=0)

    time_steps = np.arange(1, num_steps + 1) / 10.0
    agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for agent_idx in range(num_agents):
        axes[0].plot(time_steps, min_mean[:, agent_idx],
                     label=f'Agent {agent_idx+1}', linewidth=2, marker='o', markersize=4,
                     color=agent_colors[agent_idx], alpha=0.7)
    axes[0].plot(time_steps, bon_min_mean, label='BoN Min', linewidth=2, marker='s', markersize=6, color='black', linestyle='--')
    axes[0].set_xlabel('Generation Portion', fontsize=11)
    axes[0].set_ylabel('Similarity', fontsize=11)
    axes[0].set_title('Min Similarity (Row-wise)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, ncol=2)

    for agent_idx in range(num_agents):
        axes[1].plot(time_steps, max_mean[:, agent_idx],
                     label=f'Agent {agent_idx+1}', linewidth=2, marker='s', markersize=4,
                     color=agent_colors[agent_idx], alpha=0.7)
    axes[1].plot(time_steps, bon_max_mean, label='BoN Max', linewidth=2, marker='s', markersize=6, color='black', linestyle='--')
    axes[1].set_xlabel('Generation Portion', fontsize=11)
    axes[1].set_ylabel('Similarity', fontsize=11)
    axes[1].set_title('Max Similarity (Row-wise)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, ncol=2)

    for agent_idx in range(num_agents):
        axes[2].plot(time_steps, mean_mean[:, agent_idx],
                     label=f'Agent {agent_idx+1}', linewidth=2, marker='^', markersize=4,
                     color=agent_colors[agent_idx], alpha=0.7)
    axes[2].plot(time_steps, bon_mean_mean, label='BoN Mean', linewidth=2, marker='s', markersize=6, color='black', linestyle='--')
    axes[2].set_xlabel('Generation Portion', fontsize=11)
    axes[2].set_ylabel('Similarity', fontsize=11)
    axes[2].set_title('Mean Similarity (Row-wise)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(f'out/{exp_name}/similarity_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
