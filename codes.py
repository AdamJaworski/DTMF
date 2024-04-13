import numpy as np
import global_variables

keys = {
    (697, 1209): '1',
    (697, 1336): '2',
    (697, 1477): '3',
    (770, 1209): '4',
    (770, 1336): '5',
    (770, 1477): '6',
    (852, 1209): '7',
    (852, 1336): '8',
    (852, 1477): '9',
    (941, 1209): '*',
    (941, 1336): '0',
    (941, 1477): '#'
}


def get_freq(low_f, high_f) -> str:
    low_v  = np.array([697, 770, 852, 941])
    high_v = np.array([1209, 1336, 1477])

    low_min = abs(low_v - low_f)
    high_min = abs(high_v - high_f)

    if np.min(low_min) > global_variables.FREQ_TOLERANCE:
        return ''
    if np.min(high_min) > global_variables.FREQ_TOLERANCE:
        return ''

    low_f = low_v[np.where(low_min == np.min(low_min))][0]
    high_f = high_v[np.where(high_min == np.min(high_min))][0]

    return keys[(low_f, high_f)]


def extract_number(audio, fs) -> str:
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

    if low_f and high_f:
        return get_freq(low_f, high_f)
    else:
        return ''