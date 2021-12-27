from get_traces import get_traces
from profiling_ensemble_bpelm import ENSEMBLE_BPELM
from cae import CAE


if __name__ == "__main__":
    # For all subkeys in ASCAD raw data.
    IoI = {0: [61000, 62024],
           1: [64000, 65024],
           2: [46500, 47524],
           3: [34000, 35024],
           4: [48500, 49524],
           5: [41000, 42024],
           6: [38100, 39124],
           7: [36000, 37024],
           8: [27700, 28724],
           9: [40000, 41024],
           10: [30000, 31024],
           11: [44400, 45424],
           12: [21400, 22424],
           13: [23600, 24624],
           14: [50000, 51024],
           15: [19700, 20724]}
    model_path = './models/auto_encoder_model'
    data_path = './ASCAD/ATMega8515_raw_traces.h5'
    traces, pt, key = get_traces(data_path, [0, 100000], 100, 1)
    cae = CAE(IoI)
    traces_profiling = traces[:50000]
    pt_profiling = pt[:50000]
    key_profiling = key[0]

    traces_attack = traces[50000:]
    pt_attack = pt[50000:]
    key_focus = key[0]

    ########### Keep the max and min same as training. ###########


    for i_byte in range(2,3):
        tmp_traces_profiling = cae.encoder_preprocessing(traces_profiling, model_path, i_byte).reshape([len(traces_profiling), -1])
        tmp_traces_attack = cae.encoder_preprocessing(traces_attack, model_path, i_byte).reshape([len(traces_attack), -1])
        caf_elm = ENSEMBLE_BPELM(elm_nodes=1024, bp_epochs=10, bp_lr=[1, 0.1], bp_cp=1, elm_alpha=1, elm_weight_range=[-1, 1],
                                 elm_bias_range=[0, 1], elm_use_bias=True)

        min_trace_number, guess_key, ranks, training_time = \
            caf_elm.train_and_attack(X_profiling=tmp_traces_profiling, pt_profiling=pt_profiling,
                                     key_profiling=key_profiling[i_byte], X_attack=tmp_traces_attack,
                                     pt_attack=pt_attack, focus_key=key_focus[i_byte], byte=i_byte, num_traces=5000,
                                     n_model=3)