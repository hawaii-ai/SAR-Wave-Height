import numpy as np
import h5py
import math
from tensorflow.keras.utils import Sequence

class SARGenerator(Sequence):
    """
    Generator to be passed to fit_generator, predict_generator, etc. using Keras interface.
    
    Arguments:
    filename (string) -- path to h5py file containing data
    dataset (string) -- [train, val, test] for which to pull out data
    batch_size (int) -- number of data points to read out at a time. (last read may have less than batch size)
    """
    def __init__(self, filename, split='2015_2016', batch_size=100):
        self.filename = filename
        self.split = split
        self.h5file = h5py.File(filename, 'r')
        self.data = self.h5file[split]
        self.batch_size = batch_size
        self.irange = (0, self._num_examples()) # Range to return.
                

    def __del__(self):
        self.h5file.close()
        
    def _num_examples(self):
        """Number of notebooks in file."""
        return self.data['spectrum'].shape[0]
        
    
#     def _get_example(self, idx):
#         """Return single example."""
#         pass
    
#     def _get_batch(self, idx):
#         """Return batch from _get_example."""
#         pass
    
    def __len__(self):
        """Number of batches. Not all batches must be full."""
        return math.ceil((self.irange[1] - self.irange[0]) / self.batch_size)

    def __getitem__(self, idx):
        """Return batch."""
        return self._get_batch_contiguous(idx)
    
    def _get_batch_contiguous(self, idx):
        """Return batch contiguous. This will be faster, but hard to shuffle data."""
        start = self.irange[0] + self.batch_size * idx
        stop = np.minimum(start + self.batch_size, self._num_examples())
        
        # Image spectra.
        spectrum = self.data['spectrum'][start:stop]
        assert spectrum.shape[1:] == (72, 60, 2)
        assert not np.any(np.isnan(spectrum))
        #assert not np.any(spectrum > 10000), spectrum.max()
        spectrum[spectrum > 100000] = 0
        
        # High level features. Should be preprocessed already.
        names = ['cwave', 'dxdt', 'latlonSARcossin', 'todSAR', 'incidence', 'satellite']
        features = []
        for name in names:
            if name in self.data:
                temp = self.data[name][start:stop]
            elif name == 'dxdt':
                temp = np.zeros_like(self.data['incidence'][start:stop])
            else:
                raise Exception
            features.append(temp)
        features = np.hstack(features)
        #features = np.hstack([self.data[name][start:stop] for name in names])
        assert features.shape[1] == 32, features.shape
        assert not np.any(np.isnan(features))
        assert not np.any(features > 1000), features.max()
        
        # Target in m. 
        if 'hsALT' in self.data:
            target = self.data['hsALT'][start:stop]
            assert target.shape[1] == 1
            assert not np.any(np.isnan(target))
            assert not np.any(target > 100), target
        else:
            target = None
        
        inputs = [spectrum, features]
        outputs = target
        return inputs, outputs
    
class SARGenerator2(Sequence):
    """
    Generator to be passed to fit_generator, predict_generator, etc. using Keras interface.
    
    Arguments:
    filename (string) -- path to h5py file containing data
    dataset (string) -- [train, val, test] for which to pull out data
    batch_size (int) -- number of data points to read out at a time. (last read may have less than batch size)
    """
    def __init__(self, filename, groups=['2015_2016'], batch_size=100):
        self.filename = filename
        self.groups = groups
        self.h5file = h5py.File(filename, 'r')
        self.batch_size = batch_size
        #self.irange = (0, self._num_examples()) # Range to return.
        self._calc_batches_per_group()

    def __del__(self):
        self.h5file.close()
        
    def _num_examples_group(self, group):
        """Number of notebooks in file."""
        return self.h5file[group]['spectrum'].shape[0]
    
    def _calc_batches_per_group(self):
        """
        Calc num batches per group. Also an index of which batch comes from which group.
        """
        batches_per_group = {}
        for group in self.groups:
            num_group = self.h5file[group]['spectrum'].shape[0]
            batches_per_group[group] = math.ceil(num_group / self.batch_size)
        self.batches_per_group = batches_per_group
        
        # Total number of batches.
        num_batches_total = 0
        for g, num_batches in batches_per_group.items():
            num_batches_total += num_batches
        self.num_batches_total = num_batches_total
        
        # 
        self.idx2igroup = np.zeros((num_batches_total, ), dtype=int)
        self.idx2inbatchidx = np.zeros((num_batches_total, ), dtype=int) # Within-group index.
        self.igroup2group = {}
        count = 0
        for i, (g, n) in enumerate(batches_per_group.items()):
            self.igroup2group[i] = g
            self.idx2igroup[count:count+n] = i
            self.idx2inbatchidx[count:count+n] = np.arange(0, n, dtype=int)
        
        return
    
    def __len__(self):
        """Number of batches. Not all batches must be full."""
        return self.num_batches_total

    def __getitem__(self, idx):
        """Return batch."""
        return self._get_batch_contiguous(idx)
    
    def _get_batch_contiguous(self, idx):
        """Return batch contiguous. This will be faster, but hard to shuffle data."""
        group = self.igroup2group[self.idx2igroup[idx]]
        start = self.batch_size * self.idx2inbatchidx[idx]
        stop = np.minimum(start + self.batch_size, self._num_examples_group(group))
        dataset = self.h5file[group]
        
        # Image spectra.
        spectrum = dataset['spectrum'][start:stop]
        assert spectrum.shape[1:] == (72, 60, 2)
        assert not np.any(np.isnan(spectrum))
        #assert not np.any(spectrum > 10000), spectrum.max()
        spectrum[spectrum > 100000] = 0
        
        # High level features. Should be preprocessed already.
        names = ['cwave', 'dxdt', 'latlonSARcossin', 'todSAR', 'incidence', 'satellite']
        features = []
        for name in names:
            if name in dataset:
                temp = dataset[name][start:stop]
            elif name == 'dxdt':
                temp = np.zeros_like(dataset['incidence'][start:stop])
            else:
                raise Exception
            features.append(temp)
        features = np.hstack(features)
        #features = np.hstack([self.data[name][start:stop] for name in names])
        assert features.shape[1] == 32, features.shape
        assert not np.any(np.isnan(features))
        assert not np.any(features > 1000), features.max()
        
        # Target in m. 
        if 'hsALT' in dataset:
            target = dataset['hsALT'][start:stop]
            assert target.shape[1] == 1
            assert not np.any(np.isnan(target))
            assert not np.any(target > 100), target
        else:
            target = None
        
        inputs = [spectrum, features]
        outputs = target
        return inputs, outputs