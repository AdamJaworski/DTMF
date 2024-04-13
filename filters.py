from scipy.signal import butter, filtfilt
import numpy as np
import utilities
import global_variables

order_ = 8
gain_ = 2


def butter_highpass(cutoff, fs, order=order_):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=order_):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def highpass_filter(data, cutoff, fs, order=order_, gain=gain_):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y * gain


def lowpass_filter(data, cutoff, fs, order=order_, gain=gain_):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y * gain


def select_freq(audio, fs) -> np.ndarray:
    target_freq = [697, 770, 852, 941, 1209, 1336, 1477]
    audio_list  = []
    for freq in target_freq:
        y = highpass_filter(audio, freq - global_variables.FREQ_TOLERANCE, fs)
        y = lowpass_filter(y, freq + global_variables.FREQ_TOLERANCE, fs)
        y = utilities.normalize_audio(y)
        audio_list.append(y)
    output_audio = None
    for audio_ in audio_list:
        if output_audio is None:
            output_audio = audio_
        output_audio += audio_
    return output_audio
