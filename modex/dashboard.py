#!/usr/bin/env python3
"""
Dashboard utilities for visualizing debate results and accuracy trends.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
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
        """Return True if the stream is a TTY (for tqdm compatibility)."""
        return self.stream.isatty()


def display_results(args, all_results, exp_name):

    scores = np.array([result['scores'] for result in all_results])
    bon_scores = np.array([result['bon_scores'] for result in all_results]).mean(axis=0)
    single_agent_scores_raw = np.array([result['single_agent_scores'] for result in all_results])

    one_dim_scores = False
    if len(scores.shape) == 1:
        one_dim_scores = True
        scores = np.expand_dims(scores, axis=-1)
        single_agent_scores_raw = np.expand_dims(single_agent_scores_raw, axis=-1)

    avg_scores = np.mean(scores, axis=0)

    single_agent_scores = np.mean(single_agent_scores_raw, axis=0)
    single_mean, single_std = np.mean(single_agent_scores, axis=0), np.std(single_agent_scores, axis=0)

    # p_value: paired t-test over samples (voted score vs. mean single-agent score)
    p_values = []
    for metric_idx in range(avg_scores.shape[0]):
        voted_series = scores[:, metric_idx]                              # (num_samples,)
        baseline_series = single_agent_scores_raw[:, :, metric_idx].mean(axis=1)  # (num_samples,)
        _, p_val = ttest_rel(voted_series, baseline_series, nan_policy='omit')
        p_values.append(p_val)
    p_values = np.array(p_values)

    depths = np.array([result['depth'] for result in all_results])

    if args.data in args.closed_ended_tasks + args.math_reasoning_tasks:
        mv_scores = np.array([result['mv_scores'] for result in all_results], dtype=int)
        if one_dim_scores:
            mv_scores = np.expand_dims(mv_scores, axis=-1)

        print(f"Accuracy: {avg_scores[0]*100:.2f}  ||  Single Agent: {single_mean[0]*100:.2f} ± {single_std[0]*100:.2f}  ||  Majority Vote: {mv_scores.mean()*100:.2f}")
        print(f"p_value: Accuracy={p_values[0]:.4f}")
        print(f"Average Depth: {depths.mean():.1f} ± {depths.std():.1f}\n\n")
        
        for i in range(len(single_agent_scores)):
            print(f"Agent {i+1}: Accuracy={single_agent_scores[i][0]*100:.2f}")
        
        print(f"Best of N: {bon_scores*100:.2f}")
        
        # PLOT
        # Compute cumulative means for trajectories
        num_samples = len(all_results)
        time_steps = np.arange(1, num_samples + 1)
        
        # Trajectory of voted response scores (cumulative mean)
        voted_cumulative = np.cumsum(scores, axis=0) / time_steps[:, np.newaxis]
        
        # Trajectory of each agent's scores (cumulative mean)
        num_agents = single_agent_scores_raw.shape[1]
        agent_cumulative = np.zeros((num_samples, num_agents + 1, 1))
        for agent_idx in range(num_agents):
            agent_scores = single_agent_scores_raw[:, agent_idx, :]  # (num_samples, 1)
            agent_cumulative[:, agent_idx, :] = np.cumsum(agent_scores, axis=0) / time_steps[:, np.newaxis]
        agent_cumulative[:, agent_idx+1, :] = np.cumsum(mv_scores, axis=0) / time_steps[:, np.newaxis]

        metric_names = ['Accuracy']
        colors = ['#1f77b4']
        agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents + 1))

        fig, axes = plt.subplots(1, 1, figsize=(6, 5))

        for metric_idx, (metric_name, color) in enumerate(zip(metric_names, colors)):
            axes.plot(time_steps, voted_cumulative[:, metric_idx],
                      color=color, linewidth=3, label='Voted Response', linestyle='-', marker='o', markersize=4)
            for agent_idx in range(num_agents):
                axes.plot(time_steps, agent_cumulative[:, agent_idx, metric_idx],
                          color=agent_colors[agent_idx], linewidth=2,
                          label=f'Agent {agent_idx + 1}', alpha=0.7, linestyle='--')
            axes.plot(time_steps, agent_cumulative[:, agent_idx+1, metric_idx],
                      color='magenta', linewidth=2, label='Majority Vote', alpha=0.7, linestyle='--')
            axes.set_xlabel('Sample Number', fontsize=11)
            axes.set_ylabel(f'{metric_name} Score', fontsize=11)
            axes.set_title(f'{metric_name} Trajectory', fontsize=12, fontweight='bold')
            axes.grid(True, alpha=0.3)
            axes.legend(fontsize=9)

    elif args.data in args.code_generation_tasks: # coding tasks
        print(f"Pass@1: {avg_scores[0]*100:.2f}  ||  Single Agent: {single_mean[0]*100:.2f} ± {single_std[0]*100:.2f}")
        print(f"BLEU: {avg_scores[1]*100:.2f}  ||  Single Agent: {single_mean[1]*100:.2f} ± {single_std[1]*100:.2f}")
        print(f"p_value: Pass@1={p_values[0]:.4f}  ||  BLEU={p_values[1]:.4f}")
        print(f"Average Depth: {depths.mean():.1f} ± {depths.std():.1f}\n\n")
        
        for i in range(len(single_agent_scores)):
            print(f"Agent {i+1}: Pass@1={single_agent_scores[i][0]*100:.2f}  ||  BLEU={single_agent_scores[i][1]*100:.2f}")
        
        print(f"Best of N: Pass@1={bon_scores[0]*100:.2f}  ||  BLEU={bon_scores[1]*100:.2f}")
        
        # PLOT
        # Compute cumulative means for trajectories
        num_samples = len(all_results)
        time_steps = np.arange(1, num_samples + 1)
        
        # Trajectory of voted response scores (cumulative mean)
        voted_cumulative = np.cumsum(scores, axis=0) / time_steps[:, np.newaxis]
        
        num_agents = single_agent_scores_raw.shape[1]
        agent_cumulative = np.zeros((num_samples, num_agents, 2))
        for agent_idx in range(num_agents):
            agent_scores = single_agent_scores_raw[:, agent_idx, :]
            agent_cumulative[:, agent_idx, :] = np.cumsum(agent_scores, axis=0) / time_steps[:, np.newaxis]

        metric_names = ['Pass@1', 'BLEU']
        colors = ['#1f77b4', '#ff7f0e']
        agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for metric_idx, (metric_name, color) in enumerate(zip(metric_names, colors)):
            axes[metric_idx].plot(time_steps, voted_cumulative[:, metric_idx],
                                  color=color, linewidth=3, label='Voted Response', linestyle='-', marker='o', markersize=4)
            for agent_idx in range(num_agents):
                axes[metric_idx].plot(time_steps, agent_cumulative[:, agent_idx, metric_idx],
                                      color=agent_colors[agent_idx], linewidth=2,
                                      label=f'Agent {agent_idx + 1}', alpha=0.7, linestyle='--')
            axes[metric_idx].set_xlabel('Sample Number', fontsize=11)
            axes[metric_idx].set_ylabel(f'{metric_name} Score', fontsize=11)
            axes[metric_idx].set_title(f'{metric_name} Trajectory', fontsize=12, fontweight='bold')
            axes[metric_idx].grid(True, alpha=0.3)
            axes[metric_idx].legend(fontsize=9)

    elif args.data in args.text_generation_tasks: # generative task in general
        print(f"ROUGE 1: {avg_scores[0]*100:.2f}  ||  Single Agent: {single_mean[0]*100:.2f} ± {single_std[0]*100:.2f}")
        print(f"ROUGE 2: {avg_scores[1]*100:.2f}  ||  Single Agent: {single_mean[1]*100:.2f} ± {single_std[1]*100:.2f}")
        print(f"ROUGE L: {avg_scores[2]*100:.2f}  ||  Single Agent: {single_mean[2]*100:.2f} ± {single_std[2]*100:.2f}")
        print(f"BLEU: {avg_scores[3]*100:.2f}  ||  Single Agent: {single_mean[3]*100:.2f} ± {single_std[3]*100:.2f}")
        print(f"p_value: R1={p_values[0]:.4f}  ||  R2={p_values[1]:.4f}  ||  RL={p_values[2]:.4f}  ||  BLEU={p_values[3]:.4f}")
        print(f"Average Depth: {depths.mean():.1f} ± {depths.std():.1f}\n\n")
        
        for i in range(len(single_agent_scores)):
            print(f"Agent {i+1}: R1={single_agent_scores[i][0]*100:.2f} || R2={single_agent_scores[i][1]*100:.2f} || RL={single_agent_scores[i][2]*100:.2f} || BLEU={single_agent_scores[i][3]*100:.2f}")
        
        print(f"Best of N: R1={bon_scores[0]*100:.2f} || R2={bon_scores[1]*100:.2f} || RL={bon_scores[2]*100:.2f} || BLEU={bon_scores[3]*100:.2f}")
        
        # PLOT
        # Compute cumulative means for trajectories
        num_samples = len(all_results)
        time_steps = np.arange(1, num_samples + 1)
        
        # Trajectory of voted response scores (cumulative mean)
        voted_cumulative = np.cumsum(scores, axis=0) / time_steps[:, np.newaxis]
        
        num_agents = single_agent_scores_raw.shape[1]
        agent_cumulative = np.zeros((num_samples, num_agents, 4))
        for agent_idx in range(num_agents):
            agent_scores = single_agent_scores_raw[:, agent_idx, :]
            agent_cumulative[:, agent_idx, :] = np.cumsum(agent_scores, axis=0) / time_steps[:, np.newaxis]

        metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        for metric_idx, (metric_name, color) in enumerate(zip(metric_names, colors)):
            axes[metric_idx].plot(time_steps, voted_cumulative[:, metric_idx],
                                  color=color, linewidth=3, label='Voted Response', linestyle='-', marker='o', markersize=4)
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
    
    # Convert nested lists to numpy arrays
    # Each trajectory is a list of arrays (one per time step), each array has num_agents elements
    num_samples = len(similarity_trajectories['min'])
    if num_samples == 0:
        return
    
    # Get number of time steps and agents from first sample
    num_steps = len(similarity_trajectories['min'][0])
    num_agents = len(similarity_trajectories['min'][0][0]) if num_steps > 0 else 0
    
    if num_steps == 0 or num_agents == 0:
        return
    
    # Convert to numpy arrays: shape (num_samples, num_steps, num_agents)
    # Each sample is a list of arrays (one per time step)
    min_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['min']])
    bon_min_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['bon_min']])
    max_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['max']])
    bon_max_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['bon_max']])
    mean_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['mean']])
    bon_mean_trajectories = np.array([np.array(sample) for sample in similarity_trajectories['bon_mean']])
    
    # Compute mean across all samples for each time step and agent
    # Shape: (num_steps, num_agents)
    min_mean = np.mean(min_trajectories, axis=0)
    bon_min_mean = np.mean(bon_min_trajectories, axis=0)
    max_mean = np.mean(max_trajectories, axis=0)
    bon_max_mean = np.mean(bon_max_trajectories, axis=0)
    mean_mean = np.mean(mean_trajectories, axis=0)
    bon_mean_mean = np.mean(bon_mean_trajectories, axis=0)
    
    # Create time steps (x-axis)
    time_steps = np.arange(1, num_steps + 1) / 10.0
    
    # Create three subplots in a horizontal row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get colors for each agent
    agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Plot Min similarity in first subplot
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
    
    # Plot Max similarity in second subplot
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
    
    # Plot Mean similarity in third subplot
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