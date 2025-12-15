import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from tabulate import tabulate
import seaborn as sns

# function to transform indices in one-hot coded vectors
def one_hot(index, vocabulary_size):
    input = [0] * vocabulary_size
    input[index] = 1
    return input

# sigmoid activation function
def sigmoid_activation(data):
    data_clipped = np.clip(data, -50, 50)
    activated_hidden_unit = 1/(1+np.exp(-data_clipped))
    return activated_hidden_unit

# softmax activation function
def softmax_activation(data):
    if data.ndim == 1:
        data = data[np.newaxis, :]
    exp_data = np.exp(data - np.max(data, axis = 1, keepdims = True))
    return exp_data/(np.sum(exp_data, axis = 1, keepdims = True))

# cross entropy loss calculation
def cross_entropy(prediction, actual_next_word):
    loss = -np.sum((actual_next_word * np.log(prediction + 1e-8)), axis=1) # (batch_size, )
    return loss

# first word selection for language generation
def select_first_word(first_words):
    first_word = random.choice(first_words)
    # return one randomly drawn word
    return first_word

# translation from number to word
def translate_prediction(word, unique_words):
    # reverse the unique_words dictionary
    numbers_to_words = {v: k for k, v in unique_words.items()}
    # translate the given word and return it
    translated_word = numbers_to_words[word]
    return translated_word

# display scatterplot of distribution
def show_dis(vector):
    vector = np.array(vector).flatten()
    plt.scatter(range(len(vector)), vector)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# set working directory at script home
def set_wd():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

# split file name to extract parts from it
def split_file_name(file_name):
    basename = os.path.basename(file_name)
    name, _ = os.path.splitext(basename)
    split = name.split('_')
    return split

def check_right_winner(row):
    winner_diane = row['winner Diane']
    top5 = row['top 5 internal prediction']
    prediction = row['predicted index']
    probs = row['top 5 probabilities']
    
    if row['confidence']:
        if top5[winner_diane] == prediction:
            return True
        else:
            return False
    else:
        if winner_diane == np.argmin(probs):
            return True
        else:
            return False
        
def check_iteration_difference(df):
    stats = (df.groupby('condition')['iteration count']
        .median()
        .reindex(['match', 'no match low conf', 'no match high conf']))
    no_missing = stats.notna().all()
    increasing = stats.is_monotonic_increasing
    unique = stats.is_unique
            
    # check for big enough difference
    high = stats.get('no match high conf')
    low = stats.get('no match low conf')

    eps = 1e-12
    if pd.notna(high) and (high > eps) and pd.notna(low):
        effect_pct = (high - low) / high * 100.0
    else:
        effect_pct = np.nan

    threshold_pct = 5.0
    effect_ok = pd.notna(effect_pct) and (effect_pct >= threshold_pct)

    if no_missing and increasing and unique and effect_ok:
        return True
    else:
        return False
    
def prepare_df(df):
    # drop nans
    df_no_nan = df.dropna().copy()
    # set conditions
    df_no_nan['condition'] = 'other'
    df_no_nan.loc[df['match'] == True, 'condition'] = 'match'
    df_no_nan.loc[
        (df_no_nan['match'] == False) & (df_no_nan['confidence'] == True), 'condition'
    ] = 'no match high conf'
    df_no_nan.loc[
        (df_no_nan['match'] == False) & (df_no_nan['confidence'] == False), 'condition'
    ] = 'no match low conf'
    # set data types
    df_no_nan = df_no_nan.astype({
        'iteration count': int,
        'winner Diane': int,
        'match': bool,
        'confidence': bool,
        'internal winner': int
    })
    # combine bias and h/t and drop original columns
    df_no_nan['bias'] = df_no_nan['bias Teresa'].astype(str) + '/' + df_no_nan['bias Bob'].astype(str)
    df_no_nan['h/t'] = df_no_nan['heard factor'].astype(str) + '/' + df_no_nan['thought factor'].astype(str)
    df_no_nan.drop(columns=['bias Teresa', 'bias Bob', 'heard factor', 'thought factor'], inplace=True)

    return df_no_nan

def keep_valid_h_t(df, valid_dict, columns):
    # make iterable tuples
    valid_h_t = set(
        (bias, h_t)
        for bias, h_t_list in valid_dict.items()
        for h_t in h_t_list
    )
    # mask df
    mask = df[columns].apply(tuple, axis=1).isin(valid_h_t)
    df_valid = df[mask].copy()

    return df_valid, valid_h_t

def summarize(file_name, data, col1, col2):
    summary = data.agg(
        median = 'median',
        q25 = lambda x: x.quantile(0.25),
        q75 = lambda x: x.quantile(0.75)).reset_index()
    summary['iqr'] = summary['q75'] - summary['q25']
    order = ['no match high conf', 'no match low conf', 'match']
    summary['condition'] = pd.Categorical(summary['condition'], categories=order, ordered=True)
    summary = summary.sort_values([col1, col2, 'condition'])

    wide_summary = summary.pivot(
        index = [col1, col2],
        columns='condition',
        values=['median', 'iqr']
    )
    wide_summary.columns = ['_'.join(col).strip() for col in wide_summary.columns.values]
    wide_summary = wide_summary.reset_index()
    col_order = [col1, col2, 'median_no match high conf', 'iqr_no match high conf', 'median_no match low conf', 'iqr_no match low conf', 'median_match', 'iqr_match']
    wide_summary = wide_summary[col_order]

    wide_summary_flat = wide_summary.copy()
    wide_summary_flat = wide_summary_flat.round(4)
    wide_summary_flat.columns = [
        '_'.join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col)
        for col in wide_summary_flat.columns
    ]

    md_table = tabulate(wide_summary_flat, headers='keys', tablefmt='github')
    with open(f'../data/{file_name}.md', 'w', encoding='utf-8') as f:
        f.write(md_table)
        f.write('\n')

    md_table_narrow = tabulate(summary, headers='keys', tablefmt='github')
    with open(f'../data/{file_name}_narrow.md', 'w', encoding='utf-8') as f:
        f.write(md_table_narrow)
        f.write('\n')

    summary.to_parquet(f'../data/{file_name}.parquet')
    wide_summary.to_parquet(f'../data/wide_{file_name}.parquet')
    
    return summary

def create_heat_map(save_path, heatmap_data, value_range, x_label, y_label, title, legend):
    outdir = os.path.dirname(save_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        heatmap_data,
        cmap="coolwarm",   # blaue Töne für negativ, rot für positiv
        center=0,          # Farbskala symmetrisch um 0 – wichtig bei [-1, 1]
        vmin=-value_range, vmax=value_range,   # Fixiere den Bereich (macht Heatmaps besser vergleichbar)
        linewidths=0.5,    # zarte Linien zwischen Zellen (optisch sauber)
        linecolor="white",
        cbar_kws={"label": legend}  # Legendenbeschriftung
        # annot=True, fmt=".2f"  # <- optional: Zahlen direkt in die Zellen schreiben
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_self_ex_interaction(row):
    dir = '../figures/interaction_plots'
    os.makedirs(dir, exist_ok=True)

    order = ['no match high conf', 'no match low conf', 'match']
    medians_standard = [row['median_no match high conf_x'], row['median_no match low conf_x'], row['median_match_x']]
    medians_other = [row['median_no match high conf_y'], row['median_no match low conf_y'], row['median_match_y']]
    standard_self_ex = row['self-excitation_x']
    standard_h_t = row['h/t_x']
    standard_h_t_save = standard_h_t.replace('/', '_')
    other_self_ex = row['self-excitation_y']
    other_h_t = row['h/t_y']
    other_h_t_save = other_h_t.replace('/', '_')
    save_path = f'{dir}/interaction_plot_{standard_self_ex}_{standard_h_t_save}_{other_self_ex}_{other_h_t_save}.png'
    
    plt.figure()
    plt.plot(order, medians_standard, marker='o', label=f'{standard_self_ex} - {standard_h_t}')
    plt.plot(order, medians_other, marker='o', label=f'{other_self_ex} - {other_h_t}')
    plt.ylabel('Median Iteration Count')
    plt.title('Median Iteration Count per Condition (novice vs improv player)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()