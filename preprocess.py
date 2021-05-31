import librosa
import numpy as np
import os, sys
import argparse
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import utils
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
import glob
from os.path import join, basename
import subprocess
import random

speaker_used = [  0, 225,   1, 226,   2, 227,   3, 228,   4, 229,   5, 230,   6,
                231,   7, 232,   8, 233,   9, 234,  10, 236,  11, 237,  12, 238,
                 13, 239,  14, 240,  15, 241,  16, 243,  17, 244,  18, 245,  19,
                246,  20, 247,  21, 248,  22, 249,  23, 250,  24, 251,  25, 252,
                 26, 253,  27, 254,  28, 255,  29, 256,  30, 257,  31, 258,  32,
                259,  33, 260,  34, 261,  35, 262,  36, 263,  37, 264,  38, 265,
                 39, 266,  40, 267,  41, 268,  42, 269,  43, 270,  44, 271,  45,
                272,  46, 273,  47, 274,  48, 275,  49, 276]


def get_spk_raw_npy_and_feats(spk_fold_path, spk, mc_dir_train, mc_dir_test, sample_rate=16000):
    spk_name = '%03d'%spk
    if spk<200:
        paths = glob.glob(join(spk_fold_path, spk_name+'*.wav'))
    else:
        paths = glob.glob(join(spk_fold_path, 'p'+spk_name+'_*.wav'))
    _ = random.shuffle(paths)
    paths = paths[0:100]
    f0s = []
    #coded_sps = []
    coded_sp_count = 0
    coded_sp_sum = np.zeros((96))
    coded_sp_sqsum = np.zeros((96))
    for wav_file in paths:
        wav_nam = basename(wav_file)
        wav = utils.load_wav(wav_file, sample_rate)
        wav = utils.wav_volume_rescaling(wav)
        wav = utils.trim_silence(wav, 30)
        f0, _, _, _, coded_sp = utils.world_encode_wav(wav, fs=sample_rate)
        if len(f0)>=256:
            f0s.append(f0)
            #coded_sps.append(coded_sp)
            coded_sp_count += np.shape(coded_sp)[0]
            coded_sp_sum += np.sum(coded_sp, axis=0)
            coded_sp_sqsum += np.sum(coded_sp**2, axis=0)
            if spk<200:
                out_name = join(mc_dir_train, spk_name+'_'+wav_nam[3:6]+'.npy')
            else:
                out_name = join(mc_dir_train, spk_name+'_'+wav_nam[5:8]+'.npy')
            np.save(out_name, coded_sp, allow_pickle=False)
            print(wav_nam, flush=True)

    log_f0_mean, log_f0_std = utils.logf0_statistics(f0s)
    #coded_sps_mean, coded_sps_std = coded_sp_stats(coded_sp_count, coded_sp_sum, coded_sp_sqsum)
    np.savez(join(mc_dir_train, spk_name+'_stats.npz'),
            log_f0_mean=log_f0_mean,
            log_f0_std=log_f0_std,
            coded_sp_count=coded_sp_count,
            coded_sp_sum=coded_sp_sum,
            coded_sp_sqsum=coded_sp_sqsum)

    return 0


def get_global_feats_and_spk_npy(mc_dir_train):
    coded_sp_count = 0
    coded_sp_sum = np.zeros((96))
    coded_sp_sqsum = np.zeros((96))
    ls = len(speaker_used)
    log_f0_means = np.zeros((ls))
    log_f0_stds = np.zeros((ls))
    for i in range(len(speaker_used)):
        spk_name = '%03d'%speaker_used[i]
        spk_stats = np.load(join(mc_dir_train, spk_name+'_stats.npz'))
        log_f0_means[i] = spk_stats['log_f0_mean']
        log_f0_stds[i] = spk_stats['log_f0_std']
        coded_sp_count += spk_stats['coded_sp_count']
        coded_sp_sum += spk_stats['coded_sp_sum']
        coded_sp_sqsum += spk_stats['coded_sp_sqsum']
    coded_sp_mean, coded_sp_std = utils.coded_sp_stats(coded_sp_count, coded_sp_sum, coded_sp_sqsum)
    np.savez(join(mc_dir_train, 'stats.npz'),
            log_f0_means=log_f0_means,
            log_f0_stds=log_f0_stds,
            coded_sp_mean=coded_sp_mean,
            coded_sp_std=coded_sp_std)

    paths = glob.glob(join(mc_dir_train, '*.npy'))
    stats = np.load(join(mc_dir_train, 'stats.npz'))
    coded_sp_mean = stats['coded_sp_mean']
    coded_sp_std = stats['coded_sp_std']
    for npy_file in paths:
        coded_sp = np.load(npy_file)
        normed_coded_sp = utils.normalize_coded_sp(coded_sp, coded_sp_mean, coded_sp_std)
        np.save(npy_file, normed_coded_sp, allow_pickle=False)
        print(npy_file, flush=True)

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type = int, default = 16000, help = "Sample rate.")
    parser.add_argument("--vctk_path", type = str, default = "/data/vctk/wav48/", help = "The original wav path to resample.")
    parser.add_argument("--zpc_path", type = str, default = "/luoshen/tts/multispeaker/wavs/", help = "The original wav path to resample.")
    parser.add_argument("--mc_dir_train", type = str, default = '/luoshen/wrb/2020/train', help = "The directory to store the training features.")
    parser.add_argument("--num_workers", type = int, default = min(cpu_count(),10), help = "The number of cpus to use.")

    argv = parser.parse_args()

    sample_rate = argv.sample_rate
    vctk_path = argv.vctk_path
    zpc_path = argv.zpc_path
    mc_dir_train = argv.mc_dir_train
    num_workers = argv.num_workers

    os.makedirs(mc_dir_train, exist_ok=True)
    print("number of workers: ", num_workers)

    '''
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for i in range(len(speaker_used)):
        spk = speaker_used[i]
        if spk<200: # zpc
            spk_path = zpc_path
        else:
            spk_path = os.path.join(vctk_path, 'p%03d'%spk)
        futures.append(executor.submit(partial(get_spk_raw_npy_and_feats, spk_path, spk, mc_dir_train, sample_rate)))
    result_list = [future.result() for future in futures]
    print(result_list)
    '''
    a = get_global_feats_and_spk_npy(mc_dir_train)

    sys.exit(0)

