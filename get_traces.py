#-- coding: utf-8 --
#@Time : 2021/5/28 23:44
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@Software: PyCharm


import h5py
import os
import numpy as np


def get_traces(metadata_path, target_range, desync=0, seed=1):
    """

    :param metadata_path: A string of data path.
    :param target_range: A list including start and end of output range.
    :param desync: An integer of stimulate jitter.
    :param seed: An integer of seed.
    :return: traces, plaintexts, keys
    """
    target_range_start = target_range[0]
    target_range_stop = target_range[1]
    if 'h5' in os.path.split(metadata_path)[-1]:
        np.random.seed(seed)
        random_shift = (np.random.random([60000]) * desync).astype(np.int32)
        in_file = h5py.File(metadata_path, "r")

        if desync == 0 or 'raw' not in metadata_path:
            try:
                traces = np.array(in_file['traces'][:, target_range_start: target_range_stop])
                plaintext = np.array(in_file['metadata']['plaintext'])
                key = np.array(in_file['metadata']['key'])
            except KeyError:
                profiling_traces = np.array(in_file['Profiling_traces']['traces'])
                attack_traces = np.array(in_file['Attack_traces']['traces'])
                traces = np.concatenate([profiling_traces, attack_traces], axis=0)
                del profiling_traces, attack_traces
                profiling_plaintext = np.array(in_file['Profiling_traces']['metadata']['plaintext'])
                attack_plaintext = np.array(in_file['Attack_traces']['metadata']['plaintext'])
                plaintext = np.concatenate([profiling_plaintext, attack_plaintext], axis=0)
                del profiling_plaintext, attack_plaintext
                profiling_key = np.array(in_file['Profiling_traces']['metadata']['key'])
                attack_key = np.array(in_file['Attack_traces']['metadata']['key'])
                key = np.concatenate([profiling_key, attack_key], axis=0)
                key = np.concatenate([key[:, 2:3] for i in range(16)], axis=1)
                del profiling_key, attack_key

        else:
            traces = in_file['traces']
            target_range_start = np.maximum(target_range_start, desync//2)
            target_range_stop = np.minimum(target_range_stop, len(traces[0]) - desync//2)
            traces = traces[:, target_range_start - desync//2: target_range_stop + desync//2]
            traces = np.array(traces)
            for i in range(len(traces)):
                traces[i, 0:target_range_stop - target_range_start] = \
                    traces[i, random_shift[i]:target_range_stop - target_range_start + random_shift[i]]
            traces = traces[:, 0:target_range_stop - target_range_start]
            plaintext = np.array(in_file['metadata']['plaintext'])
            key = np.array(in_file['metadata']['key'])
        np.random.seed(None)
        return traces, plaintext, key

    elif 'npy' in os.path.split(metadata_path)[-1]:
        print('loading training data')
        traces = np.load(metadata_path)
        with open(os.path.join(os.path.split(metadata_path)[0], 'plaintext.txt')) as file:
            plaintexts = file.readlines()
        plaintexts = [[int('0x' + i, 16) for i in item.strip().split(' ')] for item in plaintexts]
        plaintexts = np.array(plaintexts)
        with open(os.path.join(os.path.split(metadata_path)[0], 'keys.txt')) as file:
            keys = file.readlines()
        keys = [[int('0x' + i, 16) for i in item.strip().split(' ')] for item in keys]
        keys = np.array(keys)
        if desync == 0:
            traces = traces[:, target_range_start:target_range_stop]
            pass
        else:
            target_range_start = np.maximum(target_range_start, desync//2)
            target_range_stop = np.minimum(target_range_stop, len(traces[0]) - desync//2)
            traces = traces[:, target_range_start - desync//2: target_range_stop + desync//2]
            np.random.seed(seed)
            random_shift = (np.random.random([len(traces)]) * desync).astype(np.int32)
            for i in range(len(traces)):
                traces[i, 0:target_range_stop - target_range_start] = \
                    traces[i, random_shift[i]:target_range_stop - target_range_start + random_shift[i]]
            traces = traces[:, 0:target_range_stop - target_range_start]

        np.random.seed(None)
        return traces, plaintexts, keys
    else:
        raise NameError
