#-- coding: utf-8 --
#@Time : 2021/5/16 23:31
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@Software: PyCharm


import sys
import numpy as np
from rvrr import RVRR
import time
import matplotlib.pyplot as plt


AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

class ENSEMBLE_BPELM:
    def __init__(self, elm_nodes, bp_epochs, bp_lr=list, bp_cp=1, elm_alpha=1, elm_weight_range=list,
                 elm_bias_range=list, elm_activation='relu', elm_all_standardize=False, elm_weight_init='uniform',
                 elm_links=False, elm_use_bias=False, hidden_layer_norm=False, seed=None):
        self.elm_nodes = elm_nodes
        self.bp_epochs = bp_epochs
        self.bp_lr = bp_lr
        self.bp_cp = bp_cp
        self.elm_alpha = elm_alpha
        self.elm_weight_range = elm_weight_range
        self.elm_bias_range = elm_bias_range
        self.elm_activation = elm_activation
        self.elm_all_standardize = elm_all_standardize
        self.elm_weight_init = elm_weight_init
        self.elm_links = elm_links
        self.elm_use_bias = elm_use_bias
        self.hidden_layer_norm = hidden_layer_norm
        np.random.seed(seed)

    @staticmethod
    def rank(predictions, plaintexts, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
        # Compute the rank
        if len(last_key_bytes_proba) == 0:
            # If this is the first rank we compute, initialize all the estimates to zero
            key_bytes_proba = np.zeros(256)
        else:
            # This is not the first rank we compute: we optimize things by using the
            # previous computations to save time!
            key_bytes_proba = last_key_bytes_proba

        for p in range(0, max_trace_idx-min_trace_idx):
            # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
            plaintext = plaintexts[min_trace_idx + p][target_byte]
            for i in range(0, 256):
                # Our candidate key byte probability is the sum of the predictions logs
                proba = predictions[p][AES_Sbox[plaintext ^ i]]
                if proba != 0:
                    key_bytes_proba[i] += np.log(proba)
                else:
                    # We do not want an -inf here, put a very small epsilon
                    # that correspondis to a power of our min non zero proba
                    min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                    if len(min_proba_predictions) == 0:
                        print("Error: got a prediction with only zeroes ... this should not happen!")
                        sys.exit(-1)
                    min_proba = min(min_proba_predictions)
                    key_bytes_proba[i] += np.log(min_proba**2)
        # Now we find where our real key candidate lies in the estimation.
        # We do this by sorting our estimates and find the rank in the sorted array.
        sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
        real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
        return (real_key_rank, key_bytes_proba)

    def full_ranks(self, predictions, focus_key, plaintexts, min_trace_idx, max_trace_idx, rank_step, target_byte):
        real_key = focus_key

        index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
        f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
        key_bytes_proba = []
        min_trace_number = max_trace_idx
        for t, i in zip(index, range(0, len(index))):
            real_key_rank, key_bytes_proba = self.rank(predictions[t-rank_step:t], plaintexts, real_key, t-rank_step, t, key_bytes_proba, target_byte)
            f_ranks[i] = [t - min_trace_idx, real_key_rank]
            if real_key_rank != 0:
                min_trace_number = max_trace_idx
            if real_key_rank == 0 and min_trace_number == max_trace_idx:
                min_trace_number = t

        if min_trace_number == max_trace_idx:
            min_trace_number = np.nan
        return f_ranks, min_trace_number, key_bytes_proba

    def train_and_attack(self, X_profiling, pt_profiling, key_profiling, X_attack, pt_attack, focus_key, byte, num_traces=2000, n_model=1):
        Y_profiling = AES_Sbox[pt_profiling[:, byte] ^ key_profiling]
        print('start training')
        start_time = time.time()
        predictions = np.zeros([num_traces, 256])
        X_attack = X_attack[:num_traces, :]
        models = []
        counter = 0
        while len(models) < n_model:
            models.append(RVRR(self.elm_nodes, self.elm_alpha, 256, self.elm_weight_range, self.elm_bias_range,
                               self.elm_activation, self.elm_all_standardize, self.elm_weight_init, self.elm_links,
                               self.elm_use_bias, self.hidden_layer_norm))

            models[-1].train(X_profiling, Y_profiling, self.bp_epochs, self.bp_lr, self.bp_cp)


            counter += 1
            # We test the rank over traces of the Attack dataset, with a step of 10 traces
        training_time = time.time() - start_time
        print(training_time, 'second')
        print('complete!')

        for model_ind in range(n_model):
            # Predict our probabilities
            predictions += models[model_ind].predict(X_attack, True)
        ranks, min_trace_number, key_proba = self.full_ranks(predictions, focus_key, pt_attack, 0, num_traces, 10, byte)
        guess_key = np.argmax(key_proba)
        print(f'Top 5 guess keys {np.argsort(-key_proba)[:5]}')
        # We plot the results
        x = [ranks[i][0] for i in range(0, ranks.shape[0])]
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.plot(x, y)
        plt.show()
        return min_trace_number, guess_key, ranks, training_time
