from Generator import Generator
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


def test():
    # load Teresa's training parameters for the balanced training and 64 hidden units
    directory = 'log_files'
    parameter_file = 'parameters_Teresa_50_50_64.pkl'
    path = f'{directory}/{parameter_file}'
    Teresa = Generator(path)

    # load test set (file name from pf)
    file_test_set = parameter_file.replace('parameters', 'test_set')
    path_test_set = f'../data/{file_test_set}'
    with open(path_test_set, 'rb') as f:
        test_set = pickle.load(f)

    # calculate accuracy
    accuracy, accuracy_per_word = Teresa.generate_to_test(path_test_set, teacher_forcing=False)
    accuracy_tf, accuracy_per_word_tf = Teresa.generate_to_test(path_test_set, teacher_forcing=True)
    accuracy_per_word = [float(x) for x in accuracy_per_word]
    accuracy_per_word_tf = [float(x) for x in accuracy_per_word_tf]

    loss = Teresa.train(test_set, training=False)
    perplexity = np.exp(loss)

    # create dictionary
    metrics = {
        'loss': loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'accuracy tf': accuracy_tf,
        'per word accuracy': accuracy_per_word,
        'per word accuracy tf': accuracy_per_word_tf
    }

    # save figure
    log_dir = '../figures/test_figures'
    os.makedirs(log_dir, exist_ok=True)
    basename = os.path.basename(parameter_file)
    name, _ = os.path.splitext(basename)
    figure_file_name = name.replace('parameters', 'accuracy')
    file_path = f'{log_dir}/{figure_file_name}.png'
    plt.plot(accuracy_per_word, label=f'No Teacher Forcing\nTotal: {accuracy:.2f}')
    plt.plot(accuracy_per_word_tf, label=f'Teacher Forcing\nTotal: {accuracy_tf:.2f}')
    plt.title('Top-5-Accuracy per Word')
    plt.xlabel('Word Position')
    plt.xticks([0, 1, 2, 3, 4], labels=['Word 2', 'Word 3', 'Word 4', 'Word 5', 'Word 6'])
    plt.ylabel('Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(file_path, dpi=300)
    plt.close()

    # write markdown
    lines = []
    lines.append("## Teresa's Performance Metrics with 64 hidden units")
    lines.append('')
    lines.append('| loss | perplexity | total accuracy | total accuracy with teacher forcing | per word accuracy | per word accuracy with teacher forcing |')
    lines.append('|---|---|---|---|---|---|')
    lines.append(f"| {metrics['loss']} | {metrics['perplexity']} | {metrics['accuracy']} | {metrics["accuracy tf"]} | {metrics['per word accuracy']} | {metrics["per word accuracy tf"]} |")

    md_content = '\n'.join(lines)
    with open('../data/test_metrics_Teresa_50_50_64.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

    print('Calculated and saved test metrics and figure successfully.')