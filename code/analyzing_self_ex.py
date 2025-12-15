import config_Diane
import pandas as pd
from network_functions import summarize
from network_functions import create_heat_map

def analyze_self_ex():

    standard_self_ex = config_Diane.self_excitation

    # load parquet
    sim_data = pd.read_parquet('../data/simulation_data_all_self_ex_valid.parquet')

    # extract stats for self-ex 3.5
    standard_set = sim_data.loc[sim_data['self-excitation'] == standard_self_ex]
    standard_set_grouped = standard_set.groupby(['self-excitation', 'h/t', 'condition'], observed=True)['iteration count']
    summary_standard_self_ex = summarize('summary_standard_self_ex', standard_set_grouped, 'self-excitation', 'h/t')
    possible_h_t = standard_set['h/t'].unique()
    possible_self_ex = sim_data['self-excitation'].unique()

    # create summary
    sim_data_grouped = sim_data.groupby(['self-excitation', 'h/t', 'condition'], observed=True)['iteration count']
    summary = summarize('summary_self_ex', sim_data_grouped, 'self-excitation', 'h/t')

    results_mean_diff = []
    results_match_diff_higher = []

    standard_subset = summary_standard_self_ex.loc[summary_standard_self_ex['self-excitation'] == standard_self_ex]

    for self_ex in possible_self_ex:
        subset = summary.loc[summary['self-excitation'] == self_ex]
        h_t_subset = subset['h/t'].unique()
        for h_t in possible_h_t:
            if h_t in h_t_subset:
                subsubset = subset.loc[subset['h/t'] == h_t]
                standard_subsubset = standard_subset[standard_subset['h/t'] == h_t]

                # calculate differences
                s1 = standard_subsubset['median'].reset_index(drop=True)
                s2 = subsubset['median'].reset_index(drop=True)
                diff = s2 - s1
                match_diff_higher = (list(diff)[-1] - list(diff)[0]) > 0
                mean_diff = diff.mean()

                # collect results
                result_mean_diff = {
                    'self_ex': self_ex,
                    'h_t': h_t,
                    'mean_diff': mean_diff
                }
                results_mean_diff.append(result_mean_diff)

                result_match_diff_higher = {
                    'self_ex': self_ex,
                    'h_t': h_t,
                    'match_diff_higher': int(match_diff_higher)
                }
                results_match_diff_higher.append(result_match_diff_higher)

    df_differences = pd.DataFrame(results_mean_diff)
    df_match = pd.DataFrame(results_match_diff_higher)

    heatmap_data_diff = df_differences.pivot(index='self_ex', columns='h_t', values='mean_diff')
    create_heat_map('../figures/heat_maps/heat_map_self_ex_h_t_differences.png', heatmap_data_diff, value_range=50, x_label='h/t', y_label='self-excitation', title="Sensitivity analysis: self-excitation, mean difference", legend="Δ(self-excitation, standard self-excitation)")

    heatmap_data_match = df_match.pivot(index='self_ex', columns='h_t', values='match_diff_higher')
    create_heat_map('../figures/heat_maps/heat_map_self_ex_h_t_match_higher.png', heatmap_data_match, value_range=1, x_label='h/t', y_label='self-excitation', title="Sensitivity analysis: self-excitation, match difference higher", legend="Δ(match difference, no match high confidence difference)")

    print('Created and saved heat maps successfully.')