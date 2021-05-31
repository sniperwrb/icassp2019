from torch.utils import data
import torch
import os
import random
import glob
from os.path import join, basename
import numpy as np
import sys

import time

min_length = 256   # Since we slice 256 frames from each utterance when training.
# Below is the accent info for the used 10 speakers.

speakers = [  0, 225,   1, 226,   2, 227,   3, 228,   4, 229,   5, 230,   6,
            231,   7, 232,   8, 233,   9, 234,  10, 236,  11, 237,  12, 238,
             13, 239,  14, 240,  15, 241,  16, 243,  17, 244,  18, 245,  19,
            246,  20, 247,  21, 248,  22, 249,  23, 250,  24, 251,  25, 252,
             26, 253,  27, 254,  28, 255,  29, 256,  30, 257,  31, 258,  32,
            259,  33, 260,  34, 261,  35, 262,  36, 263,  37, 264,  38, 265,
             39, 266,  40, 267,  41, 268,  42, 269,  43, 270,  44, 271,  45,
            272,  46, 273,  47, 274,  48, 275,  49, 276]

spk2idx = dict(zip(speakers, range(len(speakers))))

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, data_dir, max_load=50000):
        mc_files = glob.glob(join(data_dir, '*.npy'))
        mc_files = [i for i in mc_files] 
        _ = random.shuffle(mc_files)
        self.mc_files = self.rm_too_short_utt(mc_files, max_load=max_load)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        sys.stdout.flush()

    def rm_too_short_utt(self, mc_files, min_length=min_length, max_load=50000,
                         n_mels=None):
        new_mc_files = []
        i=0
        if n_mels is None:
            mc = np.load(mc_files[0])
            n_mels = mc.shape[1]
        ts = 0
        for mcfile in mc_files:
            #if mc.shape[0] > min_length:
            t0 = time.time()
            l = os.path.getsize(mcfile)
            t1 = time.time()
            ts += (t1-t0)
            if l > min_length*n_mels*8+128:
                new_mc_files.append(mcfile)
            i=i+1
            if (i%1000==0):
                print("Items loaded: %5d, Average time: %8.3f ms per item"%(i, ts*1000/i))
                sys.stdout.flush()
            if (i>=max_load):
                break
        return new_mc_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        mc = np.load(filename)
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        spk = int(basename(filename).split('_')[0])
        spk_idx = spk2idx[spk]
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(speakers)))  # to one-hot
        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)
        

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, src_spk=1, trg_spk=225):
        self.src_spk = '%03d'%src_spk
        self.trg_spk = '%03d'%trg_spk
        src_spk_id = spk2idx[src_spk]
        trg_spk_id = spk2idx[trg_spk]
        self.mc_files = sorted(glob.glob(join(data_dir, '{}_*.npy'.format(self.src_spk))))
        self.mc_files_trg = sorted(glob.glob(join(data_dir, '{}_*.npy'.format(self.trg_spk))))

        self.spk_stats = np.load(join(data_dir, 'stats.npz'))

        logf0_mean = self.spk_stats['log_f0_means']
        #logf0_std = self.spk_stats['log_f0_stds']
        self.logf0_mean_src = logf0_mean[src_spk_id]
        self.logf0_mean_trg = logf0_mean[trg_spk_id]
        self.mcep_mean = self.spk_stats['coded_sp_mean']
        self.mcep_std = self.spk_stats['coded_sp_std']

        if src_spk<200:
            self.src_wav_dir = '/luoshen/tts/multispeaker/wavs/'
        else:
            self.src_wav_dir = '/data/vctk/wav48/'

        self.spk_idx = trg_spk_id
        spk_cat = to_categorical([trg_spk_id], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile)
            if self.src_spk[0] in ['2', '3']:
                wavfile_path = join(self.src_wav_dir, 'p'+filename[0:3], 'p'+filename[0:3]+'_'+filename[4:7]+'.wav')
            else:
                wavfile_path = join(self.src_wav_dir, filename[0:3]+filename[4:7]+'.wav')
            batch_data.append(wavfile_path)
        return batch_data       

    

def get_loader(data_dir, batch_size=32, num_workers=1, max_load=50000):
    dataset = MyDataset(data_dir, max_load=max_load)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./data/mc/train')
    data_iter = iter(loader)
    for i in range(10):
        mc, spk_idx, acc_idx, spk_acc_cat = next(data_iter)
        print('-'*50)
        print(mc.size())
        print(spk_idx.size())
        print(acc_idx.size())
        print(spk_acc_cat.size())
        print(spk_idx.squeeze_())
        print(spk_acc_cat)
        print('-'*50)







