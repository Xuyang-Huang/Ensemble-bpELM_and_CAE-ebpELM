from get_traces import get_traces
from profiling_ensemble_bpelm import ENSEMBLE_BPELM


if __name__ == "__main__":
    # For ASCAD
    data_path = './ASCAD/ASCAD.h5'
    traces, pt, key = get_traces(data_path, [0, 700], 0)
    traces_profiling = traces[:50000]
    pt_profiling = pt[:50000]
    key_profiling = key[0][0]

    traces_attack = traces[50000:]
    pt_attack = pt[50000:]
    key_focus = key[0][0]

    i_byte = 2

    caf_elm = ENSEMBLE_BPELM(elm_nodes=128, bp_epochs=5, bp_lr=[1, 0.1], bp_cp=2, elm_alpha=1, elm_weight_range=[-1, 1],
                             elm_bias_range=[0, 1])

    min_trace_number, guess_key, ranks, training_time = \
        caf_elm.train_and_attack(X_profiling=traces_profiling, pt_profiling=pt_profiling, key_profiling=key_profiling,
                                 X_attack=traces_attack, pt_attack=pt_attack, focus_key=key_focus, byte=i_byte,
                                 num_traces=5000, n_model=3)