import argparse
from model import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename
import librosa
import utils

speakers = [  0, 225,   1, 226,   2, 227,   3, 228,   4, 229,   5, 230,   6,
            231,   7, 232,   8, 233,   9, 234,  10, 236,  11, 237,  12, 238,
             13, 239,  14, 240,  15, 241,  16, 243,  17, 244,  18, 245,  19,
            246,  20, 247,  21, 248,  22, 249,  23, 250,  24, 251,  25, 252,
             26, 253,  27, 254,  28, 255,  29, 256,  30, 257,  31, 258,  32,
            259,  33, 260,  34, 261,  35, 262,  36, 263,  37, 264,  38, 265,
             39, 266,  40, 267,  41, 268,  42, 269,  43, 270,  44, 271,  45,
            272,  46, 273,  47, 274,  48, 275,  49, 276]

spk2idx = dict(zip(speakers, range(len(speakers))))


def test(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period=16000, 96, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Restore model
    G = Generator(num_speakers=config.num_speakers).to(device)
    G_path = join(config.model_save_dir, '%d-G.ckpt'%config.resume_iters)
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    print('Loading the trained models from step %d...'%config.resume_iters)

    # Read a batch of testdata
    with open(config.src_file) as f:
        test_wavfiles_raw = f.readlines()
    test_wavfiles = []
    for i in range(len(test_wavfiles_raw)):
        s = test_wavfiles_raw[i].strip()
        if len(s)>0:
            test_wavfiles.append(s)

    spk_stats = np.load(join(config.data_dir, 'stats.npz'))
    logf0_mean = spk_stats['log_f0_means']
    mcep_mean = spk_stats['coded_sp_mean']
    mcep_std = spk_stats['coded_sp_std']
    with torch.no_grad():
        for idx in range(len(test_wavfiles)):
            wav_name = test_wavfiles[idx]
            trg_spk = '%03d'%config.trg_spk[idx]
            trg_spk_id = spk2idx[config.trg_spk[idx]]
            logf0_mean_trg = logf0_mean[trg_spk_id]
    
            spk_cat = np.zeros(len(speakers))
            spk_cat[trg_spk_id]=1.0

            wav_name = test_wavfiles[idx]
            wav, _ = librosa.load(wav_name, sr=sampling_rate, mono=True)
            wav = utils.wav_volume_rescaling(wav)
            wav = utils.trim_silence(wav, 30)
            wav = utils.wav_padding(wav, sr=sampling_rate, frame_period=frame_period, multiple = 4)  # TODO

            conds = torch.FloatTensor(spk_cat).unsqueeze_(0).to(device)

            f0, timeaxis, sp, ap = utils.world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            logf0_mean_src, _ = utils.logf0_statistics([f0])
            coded_sp = utils.world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            print("Before being fed into G: ", coded_sp.shape)

            print(np.exp(logf0_mean_src), np.exp(logf0_mean_trg))

            f0_converted = np.exp(np.ma.log(f0) - logf0_mean_src + logf0_mean_trg)
            coded_sp_norm = (coded_sp - mcep_mean) / mcep_std
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            coded_sp_converted_norm = G(coded_sp_norm_tensor, conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * mcep_std + mcep_mean
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = utils.world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                     ap=ap, fs=sampling_rate, frame_period=frame_period)

            wav_id = basename(wav_name).split('.')[0]
            librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters),
                '%s-vcto-%s.wav'%(wav_id,trg_spk)), wav_transformed, sampling_rate, norm=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('-r', '--resume_iters', type=int, default=180000, help='step to resume for testing.')
    parser.add_argument('-s', '--src_file', type=str, default='test.txt', help = 'source speaker.')
    parser.add_argument('-t', '--trg_spk', type=list, default=[13,4,243,233,14,3], help = 'target speaker.')

    parser.add_argument('--data_dir', type=str, default='/luoshen/wrb/2020/train')
    parser.add_argument('--num_speakers', type=int, default=len(speakers), help='dimension of speaker labels')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--convert_dir', type=str, default='./converted')

    config = parser.parse_args()
    
    print(config)
    test(config)
