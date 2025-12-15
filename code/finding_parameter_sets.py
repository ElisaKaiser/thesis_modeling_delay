import pandas as pd
from network_functions import plot_self_ex_interaction
from tabulate import tabulate

def find_parameter_sets():
    # import summary_self_ex
    summary = pd.read_parquet('../data/wide_summary_self_ex.parquet', columns=['self-excitation', 'h/t', 'median_no match high conf',
        'median_no match low conf', 'median_match'])

    summary_standard = summary[summary['self-excitation'] == 3.5].reset_index(drop=True)
    summary_others = summary[summary['self-excitation'] != 3.5].reset_index(drop=True)

    summary_paired = summary_standard.merge(summary_others, how='cross')

    # define minimum mean distance (10 to 15 %)
    threshold_pct = 5.0

    # look for combination that has minimum distance and diff_match > diff_no match high conf
    # find high value
    summary_paired['difference match'] = summary_paired['median_match_x'] - summary_paired['median_match_y']
    summary_paired['difference no match low conf'] = summary_paired['median_no match low conf_x'] - summary_paired['median_no match low conf_y']
    summary_paired['difference no match high conf'] = summary_paired['median_no match high conf_x'] - summary_paired['median_no match high conf_y']
    summary_paired['mean difference'] = (
        (summary_paired['difference match'] + 
        summary_paired['difference no match low conf'] + 
        summary_paired['difference no match high conf'])
        /3)
    summary_paired['effect match'] = summary_paired['difference match'] / summary_paired['median_match_x'] * 100.0
    summary_paired['effect no match low conf'] = summary_paired['difference no match low conf'] / summary_paired['median_no match low conf_x'] * 100.0
    summary_paired['effect no match high conf'] = summary_paired['difference no match high conf'] / summary_paired['median_no match high conf_x'] * 100.0
    summary_paired['effect ok'] = summary_paired[['effect match', 'effect no match low conf', 'effect no match high conf']].min(axis=1) > threshold_pct

    summary_paired['diff match higher'] = summary_paired['difference match'] > summary_paired['difference no match high conf']

    summary_paired_effect_ok = summary_paired[summary_paired['effect ok'] == True]
    summary_paired_diff_match_higher = summary_paired[summary_paired['diff match higher'] == True]

    summary_paired_eff_diff = summary_paired_effect_ok[summary_paired_effect_ok['diff match higher'] == True]

    for idx, row in summary_paired_eff_diff.iterrows():
        plot_self_ex_interaction(row)

    md_summary_paired = tabulate(summary_paired, headers='keys', tablefmt='github')
    with open('../data/summary_paired.md', 'w', encoding='utf-8') as f:
        f.write(md_summary_paired)
        f.write('\n')

    md_summary = tabulate(summary_paired_eff_diff, headers='keys', tablefmt='github')
    with open('../data/summary_paired_valid.md', 'w', encoding='utf-8') as f:
        f.write(md_summary)
        f.write('\n')

    print('Found valid parameter sets and saved Markdown table successfully.')