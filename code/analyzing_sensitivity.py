import pandas as pd
from network_functions import create_heat_map

def analyze_sensitivity():
    ### functions ###
    
    def calculate_difference(df):
        df['difference'] = df['median_no match high conf'] - df['median_no match low conf'] # positive when the difference is what I need, negative when the other way around
        #df['difference_normed'] = df['difference'] / df['difference'].abs().max()

        return df

    ### script ###
    # load summarized simulation data (valid)
    summary_valid = pd.read_parquet('../data/wide_summary.parquet')
    summary_valid_diff = calculate_difference(summary_valid)

    # create heatmap for valid bias_h/t
    heatmap_data = summary_valid_diff.pivot(index="bias", columns="h/t", values="difference")
    create_heat_map('../figures/heat_maps/heat_map_bias_h_t_valid.png', heatmap_data, value_range=100, x_label='h/t', y_label='bias', title="Sensitivity analysis: Δ(no match low conf, no match high conf)", legend="Δ(no match low conf, no match high conf)")

    # load summary all
    summary_all = pd.read_parquet('../data/wide_summary_all.parquet')
    summary_all_diff = calculate_difference(summary_all)

    # create heat map for all
    heatmap_data_all = summary_all_diff.pivot(index='bias', columns='h/t', values='difference')
    create_heat_map('../figures/heat_maps/heat_map_bias_h_t_all.png', heatmap_data_all, value_range=100, x_label='h/t', y_label='bias', title="Sensitivity analysis: Δ(no match low conf, no match high conf)", legend="Δ(no match low conf, no match high conf)")

    print('Created and saved heat maps successfully.')
