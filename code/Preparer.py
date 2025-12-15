from network_functions import one_hot
import numpy as np
import copy
import config
import pickle
import os


class MotherTeresa:
    def __init__(self, training_data: str):

        self.seq_train, seq_val, seq_test = self.prepare_training_data(training_data)
        self.input = self.batch(self.seq_train, config.batch_size)
        self.validation_set = self.batch(seq_val, config.batch_size)

        seq_test = self.batch(seq_test, config.batch_size)
        # save test set for testing
        basename = os.path.basename(training_data)
        name, ext = os.path.splitext(training_data)
        split_file_name = name.split('_')
        agent = 'Teresa' if 'Teresa' in training_data else 'Bob'
        save_file = f'../data/test_set_{agent}_{split_file_name[-2]}_{split_file_name[-1]}_{config.hidden_units}.pkl'

        with open(save_file, 'wb') as f:
            pickle.dump(seq_test, f)

        self.n_h_c = config.hidden_units
        self.n_i_o = self.vocab_size

        # Xavier initialization of weights
        self.weights_i_h = np.random.randn(self.n_i_o, self.n_h_c) * np.sqrt(2 / (self.n_i_o + self.n_h_c))
        self.weights_h_o = np.random.randn(self.n_h_c, self.n_i_o) * np.sqrt(2 / (self.n_i_o + self.n_h_c))

        # orthogonal weigh initialization
        weights_c_h = np.random.randn(self.n_h_c, self.n_h_c)
        U, _, _ = np.linalg.svd(weights_c_h, full_matrices=False)
        self.weights_c_h = U

        # bias initialization
        self.biases_h = np.zeros((1, self.n_h_c))
        self.biases_o = np.zeros((1,self.n_i_o))


    def prepare_training_data(self, training_data):
        # checking input format
        if not training_data.endswith('.txt'):
            print('Error. Input file must be a .txt file.')
            return

        # open file and reading it as a string
        with open(training_data, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # replace line breaks with spaces
        text = text.replace('\n', ' ')

        # divide text in sentences and filter out single-word sentences
        sentences = text.strip().lower().split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip() and len(sentence.split()) > 1]
        sequences = [sequence.split() for sequence in sentences] # --> list of lists with the single words

        # Split each sentence into words
        all_words = [word for sentence in sentences for word in sentence.split()]

        # Remove duplicates by converting to a set
        unique_words_set = set(all_words)

        # Sort the unique words
        sorted_unique_words = sorted(unique_words_set)

        # Create a dictionary mapping each word to an index
        self.unique_words = {word: idx for idx, word in enumerate(sorted_unique_words)}

        # translate unique words in numbers (= indices) using the dictionary
        sequences = [[self.unique_words[word] for word in sentence] for sentence in sequences]

        # extract the first words of every sentence
        self.first_words = list(set([sentence[0] for sentence in sequences]))

        # split training data in training, validation and test set
        number_of_sequences = len(sequences)
        n_train = int(number_of_sequences * config.split)
        n_rest = number_of_sequences - n_train
        n_val = int(0.5 * n_rest)

        sequences_to_train = sequences[: n_train]
        sequences_to_validate = sequences[n_train : n_train + n_val]
        sequences_to_test = sequences[n_train + n_val :]

        return sequences_to_train, sequences_to_validate, sequences_to_test


    def batch(self, data, batch_size):
        self.batch_size = batch_size

        batched_sentences = []
        sentence_batch = []

        # take sentence by sentence an put them in a batch until the batch is full
        for sentence in data:
            # put the sentence in a batch container if not full
            if len(sentence_batch) < batch_size:
                sentence_batch.append(copy.deepcopy(sentence))
            
            # if full, put the batch container in batched_sentences and in batched_mask, respectively
            else:
                batched_sentences.append(sentence_batch) # shape (batch_size, sequence_length)
                sentence_batch = []

        # if there is residue in a batch, append it as the last, shorter batch
        if sentence_batch:
            batched_sentences.append(sentence_batch)

        # replace every number/word with a one-hot coded vector
        self.vocab_size = len(self.unique_words)
        
        for batch_count, batch in enumerate(batched_sentences):
            for sentence in batch:
                for word_count, word in enumerate(sentence):
                    sentence[word_count] = one_hot(word, self.vocab_size)
            batched_sentences[batch_count] = np.array(batch).transpose(1, 0, 2) # shape (sequence_length, batch_size, vocab_size)    
        
        return batched_sentences