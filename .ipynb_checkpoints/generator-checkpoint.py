import numpy as np
import h5py
#new_means = [8.930369, 4.878463e-08]
#new_stdev = [41.090652, 6.4714637]

def gen_data(filename, dataset, batch_size):
    '''
    Generator to be passed to fit_generator, predict_generator, etc. using Keras interface.
    
    Arguments:
    filename (string) -- path to h5py file containing data
    dataset (string) -- [train, val, test] for which to pull out data
    batch_size (int) -- number of data points to read out at a time. (last read may have less than batch size)
    '''
    while True:
        with h5py.File(filename, 'r') as hf:
            curr_idx = 0
            max_idx = hf["X_"+dataset].shape[0]
            while curr_idx < max_idx:
                if curr_idx + batch_size < max_idx:
                    X = hf["X_"+dataset][curr_idx:curr_idx + batch_size]
                    X2 = hf["X2_"+dataset][curr_idx:curr_idx + batch_size]
                    yield [X, X2], hf["y_"+dataset][curr_idx:curr_idx + batch_size]
                else:
                    X = hf["X_"+dataset][curr_idx:]
                    X2 = hf["X2_"+dataset][curr_idx:]
                    yield [X, X2], hf["y_"+dataset][curr_idx:]
                curr_idx += batch_size