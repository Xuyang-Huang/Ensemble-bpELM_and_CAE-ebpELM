#-- coding: utf-8 --
#@Time : 2021/5/16 23:31
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@Software: PyCharm

from cae import CAE
from get_traces import get_traces

if __name__ == "__main__":
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

    traces, _, _ = get_traces('./ASCAD/ATMega8515_raw_traces.h5', [0, 100000], desync=100)

    cae = CAE(IoI, traces.max(), traces.min())
    cae.train(traces, model_save_path=f'./models/CAE_model.h5', epochs=200, batch_size=256)
