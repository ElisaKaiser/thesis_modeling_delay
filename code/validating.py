from Trainer import ElmanNetwork
from Generator import Generator
import config


def validate():
    results = []

    hidden_units = [32, 64, 96, 128]

    # train Teresa with different hidden units

    for hidden in hidden_units:
        result = {
        'hidden units': None,
        'loss': None,
        'epochs': None,
        'sentences': None}
        config.hidden_units = hidden
        # instantiate the Trainer with balanced training data
        save_path = f'parameters_Teresa_50_50_{hidden}'
        training_data = '../data/training_data/training_data_Teresa_50_50.txt'
        Training_Network = ElmanNetwork(save_path, training_data=training_data)
        epochs = Training_Network.train_in_epochs()
        result['hidden units'] = hidden
        result['loss'] = Training_Network.min_val_loss
        result['epochs'] = float(epochs)
        sentences = []
        Teresa = Generator(f'log_files/parameters_Teresa_50_50_{hidden}.pkl')
        for s in range(10):
            Teresa.prepare_to_generate()
            _, sentence = Teresa.generate_sentence(6)
            sentences.append(sentence)
        result['sentences'] = sentences
        results.append(result)

    # build markdown table
    lines = []
    lines.append('## Comparison of different numbers of hidden units')
    lines.append('')
    lines.append("| number of hidden units | validation loss | epochs | example sentences |")
    lines.append("|---|---|---|---|")
    for r in results:
        examples_joined = '<br>'.join(r['sentences'])
        line = f'| {r['hidden units']} | {r['loss']} | {r['epochs']} | {examples_joined} |'
        lines.append(line)

    md_content = '\n'.join(lines)

    with open('../data/validation_Teresa.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

    config.hidden_units = 64

    print('Saved Markdown file with validation loss, epochs, and example sentences successfully.')
