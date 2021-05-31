import librosa
import numpy as np
import os
import pyworld

global_eps = 1e-8

def wav_volume_rescaling(x):
    return x / np.abs(x).max() * 0.999

def trim_silence(x, top_db=60):
    x, _ = librosa.effects.trim(x, top_db=top_db, frame_length=512, hop_length=128)
    return x


def lfm(sr, n_fft, n_mels):
    melfb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels-2)
    melfb = np.concatenate((np.zeros((1,n_fft//2+1)),melfb,np.zeros((1,n_fft//2+1))),axis=0)
    max1=np.max(melfb[1,:])
    for i in range(np.argmax(melfb[1,:])):
        melfb[0,i]=max(0,max1-melfb[1,i]-melfb[2,i])
    min1=np.max(melfb[-2,:])
    for i in range(np.argmax(melfb[-2,:]),n_fft//2+1):
        melfb[-1,i]=max(0,min1-melfb[-2,i]-melfb[3,i])
    for i in range(n_mels):
        melfb[i,:]=melfb[i,:]/np.sqrt(np.sum(melfb[i,:]**2))
    melfb=melfb/np.sqrt(1.5)
    return melfb

melfb_default = lfm(sr=16000, n_fft=1024, n_mels=96)

def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav

def world_decompose(wav, fs, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim=96, eps=global_eps):
    # Get Mel-cepstral coefficients (MCEPs)
    #sp = sp.astype(np.float64)
    sp = np.log(sp + eps)
    if (fs==16000) and (dim==96):
        coded_sp = np.matmul(sp, melfb_default.T)
    else:
        melfb = lfm(sr=fs, n_fft=1024, n_mels=dim)
        coded_sp = np.matmul(sp, melfb.T)
    return coded_sp

def world_decode_spectral_envelop(coded_sp, fs, eps1=global_eps, eps2=1e-12):
    # Decode Mel-cepstral to sp
    dim = np.shape(coded_sp)[1]
    if (fs==16000) and (dim==96):
        decoded_sp = np.matmul(coded_sp, melfb_default)
    else:
        melfb = lfm(sr=fs, n_fft=1024, n_mels=dim)
        decoded_sp = np.matmul(coded_sp, melfb)
    X = np.exp(decoded_sp)-eps1
    decoded_sp = np.clip(X, eps2, np.inf)
    return decoded_sp

def world_encode_wav(wav, fs, frame_period=5.0, coded_dim=96):
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
    return f0, timeaxis, sp, ap, coded_sp

def griffin_2(stftm, hop_length=0, iters=50, center=True):
    n_fft = (np.shape(stftm)[0]-1)*2
    if (hop_length==0):
        hop_length=n_fft//4
    n_window = np.shape(stftm)[1]
    yshape = hop_length * (n_window-1) + (0 if center else n_fft)
    y = np.random.random(yshape)
    for i in range(iters):
        stftx = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, center=center)
        stftx = stftm * stftx / (np.abs(stftx) + 0.0001)
        y = librosa.core.istft(stftx, hop_length=hop_length, center=center)
    return y

def world_speech_synthesis(f0, coded_sp, ap, fs, frame_period, eps=global_eps):
    # TODO
    min_len = min([len(f0), len(coded_sp), len(ap)])
    f0 = f0[:min_len]
    coded_sp = coded_sp[:min_len]
    ap = ap[:min_len]
    eps2 = 1e-12
    decoded_sp = world_decode_spectral_envelop(coded_sp, fs, eps1=eps, eps2=eps2)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)

    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)
    return wav

def world_synthesis_data(f0s, coded_sps, aps, fs, frame_period):
    wavs = list()
    for f0, decoded_sp, ap in zip(f0s, coded_sps, aps):
        wav = world_speech_synthesis(f0, coded_sp, ap, fs, frame_period)
        wavs.append(wav)
    return wavs

def coded_sp_stats(coded_sp_count, coded_sp_sum, coded_sp_sqsum):
    # sp shape (T, D)
    coded_sps_mean = coded_sp_sum / coded_sp_count
    coded_sps_std = np.sqrt(coded_sp_sqsum / coded_sp_count - coded_sps_mean**2)
    return coded_sps_mean, coded_sps_std


def normalize_coded_sp(coded_sp, coded_sp_mean, coded_sp_std):
    normed = (coded_sp - coded_sp_mean) / coded_sp_std
    return normed


def inv_normalize_coded_sp(coded_sp, coded_sp_mean, coded_sp_std):
    normed = (coded_sp * coded_sp_std) + coded_sp_mean
    return normed


def coded_sp_padding(coded_sp, multiple = 4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)
    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded

def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src=None, mean_log_target=None, std_log_target=None):

    if mean_log_target is None:
        mean_log_target = std_log_src
    if std_log_target is None:
        f0_converted = np.exp(np.ma.log(f0) - mean_log_src + mean_log_target)
    else:
        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts


def sample_train_data(dataset_A, dataset_B, n_frames = 128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B