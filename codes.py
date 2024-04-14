import numpy as np
import global_variables
from math import floor
low_v = np.array([697, 770, 852, 941])
high_v = np.array([1209, 1336, 1477])

new_low_v  = np.array([0, 0, 0, 0])
new_high_v = np.array([0, 0, 0])


def from_dict(low_f, high_f) -> str:
    global new_low_v, new_high_v

    low_min = abs(new_low_v - low_f)
    high_min = abs(new_high_v - high_f)

    if np.min(low_min) > global_variables.FREQ_TOLERANCE:
        return ''
    if np.min(high_min) > global_variables.FREQ_TOLERANCE:
        return ''

    low_f = new_low_v[np.where(low_min == np.min(low_min))][0]
    high_f = new_high_v[np.where(high_min == np.min(high_min))][0]

    return global_variables.new_keys[(low_f, high_f)]


def get_freq(audio, fs) -> tuple:
    N = 2 ** np.ceil(np.log2(len(audio)))
    N = int(max(N, fs / 21))
    fft_result = np.fft.fft(audio, n=N)
    fft_freq = np.fft.fftfreq(N, 1 / fs)

    n = len(fft_result) // 2
    fft_magnitude = np.abs(fft_result[:n]) * 2 / len(audio)
    fft_freq = fft_freq[:n]  # Only consider the first half of the frequencies

    low_indices = np.where((fft_freq >= 680) & (fft_freq < 1000))[0]
    high_indices = np.where((fft_freq >= 1200) & (fft_freq <= 1500))[0]

    if low_indices.size > 0:
        low_f_index = low_indices[np.argmax(fft_magnitude[low_indices])]
        low_f = fft_freq[low_f_index]
    else:
        low_f = None

    if high_indices.size > 0:
        high_f_index = high_indices[np.argmax(fft_magnitude[high_indices])]
        high_f = fft_freq[high_f_index]
    else:
        high_f = None
    return high_f, low_f


def get_code(audio, fs) -> str:
    high_f, low_f = get_freq(audio, fs)
    if low_f and high_f:
        return from_dict(low_f, high_f)
    else:
        return ''


def set_keys(audio: np.ndarray, fs: float, code: str):
    global low_v, high_v
    high_f, low_f = get_freq(audio, fs)
    high_f = int(floor(high_f))
    low_f = int(floor(low_f))

    if low_f and high_f:
        low_min = abs(low_v - low_f)
        high_min = abs(high_v - high_f)

        if np.min(low_min) > global_variables.FREQ_TOLERANCE:
            return False
        if np.min(high_min) > global_variables.FREQ_TOLERANCE:
            return False

        if new_low_v[np.where(low_min == np.min(low_min))] == 0:
            new_low_v[np.where(low_min == np.min(low_min))] = low_f
        if new_high_v[np.where(high_min == np.min(high_min))] == 0:
            new_high_v[np.where(high_min == np.min(high_min))] = high_f

        if global_variables.new_keys.get((int(new_low_v[np.where(low_min == np.min(low_min))]), int(new_high_v[np.where(high_min == np.min(high_min))]))) is not None:
            return False

        global_variables.new_keys[(int(new_low_v[np.where(low_min == np.min(low_min))]), int(new_high_v[np.where(high_min == np.min(high_min))]))] = code

    return True
