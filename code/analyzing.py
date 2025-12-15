import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from tabulate import tabulate
from network_functions import keep_valid_h_t
from network_functions import summarize
from network_functions import split_file_name


def analyze():
    ### functions ###
    def save_histogram(file_path, subset):
        outdir = os.path.dirname(file_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        
        split = split_file_name(file_path)

        min_iteration_count = subset['iteration count'].min()
        max_iteration_count = subset['iteration count'].max()
        bin_size = 5
        bin_edges = [min_iteration_count]
        while bin_edges[-1] <= max_iteration_count:
            new_edge = bin_edges[-1] + bin_size
            bin_edges.append(new_edge)

        for group, sub in subset.groupby("condition"):
            plt.hist(sub["iteration count"], bins=bin_edges, alpha=0.5, label=group)

        plt.legend()
        plt.xlabel("Iteration Count")
        plt.ylabel("Frequency")
        plt.title(f"Iteration Counts by Condition (Bias: {split[-4]}/{split[-3]}, h/t: {split[-2]}/{split[-1]})")
        plt.savefig(file_path)
        plt.close()


    def save_stats(file_path, subset):
        outdir = os.path.dirname(file_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        subset = subset.copy()
        order = ['match', 'no match low conf', 'no match high conf']
        subset['condition'] = pd.Categorical(subset['condition'], categories=order, ordered=True)
        stats = subset.groupby('condition', observed=False)['iteration count'].describe()

        md_table = tabulate(stats, headers='keys', tablefmt='github')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_table)
            f.write('\n')


    def plot_interaction(file_path, subset):
        outdir = os.path.dirname(file_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        split = split_file_name(file_path)

        plt.figure(figsize=(6, 4))

        plt.plot(
            subset['condition'],
            subset['median'],
            marker = 'o'
        )

        plt.title(f'Iteration Count by Condition (Bias: {split[-4]}/{split[-3]}, h/t: {split[-2]}/{split[-1]})')
        plt.xlabel('Condition')
        plt.ylabel('Median Iteration Count')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

  
    ### script ###
    # load simulation data
    cols = ['bias', 'h/t', 'iteration count', 'condition']
    sim_data = pd.read_parquet('../data/simulation_data_processed.parquet', columns=cols)

    # make summary for whole pandas
    sim_data_grouped = sim_data.groupby(['bias', 'h/t', 'condition'], observed=True)['iteration count']
    summary_all_data = summarize('summary_all', sim_data_grouped, 'bias', 'h/t')

    # load dic with valid h/t
    with open('log_files/valid_h_t_per_bias.pkl', 'rb') as f:
        h_t_bias_combos = pickle.load(f)
    
    sim_data_valid, valid_h_t = keep_valid_h_t(sim_data, h_t_bias_combos, ['bias', 'h/t'])
    sim_data_valid.to_parquet(f'../data/simulation_data_valid.parquet')
    grouped = sim_data_valid.groupby(['bias', 'h/t', 'condition'], observed=True)['iteration count']
    summary = summarize('summary', grouped, 'bias', 'h/t')

    # create histograms and descriptive statistics markdown tables
    for bias, h_t in valid_h_t:
            
        mask = (
            (sim_data['bias'] == bias) &
            (sim_data['h/t'] == h_t)
        )
        subset = sim_data.loc[mask]

        mask = (
            (summary['bias'] == bias) &
            (summary['h/t'] == h_t)
        )
        summary_subset = summary.loc[mask]
        
        # create file paths
        save_bias = bias.replace('/', '_')
        save_h_t = h_t.replace('/', '_')
        file_path_figure = f'../figures/bias_h_t_histograms/histogram_{save_bias}_{save_h_t}.png'
        file_path_stats = f'../data/bias_h_t_stats/stats_{save_bias}_{save_h_t}.md'
        file_path_plot = f'../figures/interaction_plots/interaction_plot_{save_bias}_{save_h_t}.png'
        save_histogram(file_path=file_path_figure, subset=subset)
        save_stats(file_path=file_path_stats, subset=subset)
        plot_interaction(file_path_plot, subset=summary_subset)

    print('Saved stats, histograms, and line plots successully.')

