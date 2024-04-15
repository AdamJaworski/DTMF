import matplotlib.pyplot as plt
import numpy as np
import librosa
import global_variables
import pathlib
import sounddevice as sd
from numba import njit


def display_freq(audio: np.ndarray, fs) -> None:
    fft_result = np.fft.fft(audio)
    fft_freq = np.fft.fftfreq(len(audio), 1 / fs)

    # Taking the magnitude of the FFT result (for volume) and only the first half (due to symmetry)
    n = len(fft_result) // 2
    fft_magnitude = np.abs(fft_result[:n]) * 2 / len(audio)

    x = 650  # Lower frequency limit
    y = 1500  # Upper frequency limit
    indices = np.where((fft_freq >= x) & (fft_freq <= y))[0]
    # Plotting the Frequency vs Volume (Amplitude) graph
    plt.figure(figsize=(14, 6))
    plt.plot(fft_freq[indices], fft_magnitude[indices])
    plt.xticks(np.arange(x, y, 50))
    plt.title('Frequency vs Volume (Amplitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Volume (Amplitude)')
    plt.grid(True)
    plt.show()


def normalize_audio(audio: np.ndarray, target_peak: float = global_variables.TARGET_VOLUME) -> np.ndarray:
    """
    Normalize the audio signal so that its maximum peak is at the target peak level.
    :param audio: The input audio signal.
    :param target_peak: The target peak level.
    :return: The normalized audio signal.
    """

    max_peak = np.abs(audio).max()
    if max_peak == 0:
        return audio
    normalization_factor = target_peak / max_peak
    return audio * normalization_factor


def estimate_noise_level(audio, frame_length=2048, hop_length=512):
    energy = np.array([
        sum(abs(audio[i:i + frame_length]) ** 2)
        for i in range(0, len(audio), hop_length)
    ])

    energy_db = librosa.power_to_db(energy, ref=np.max)

    noise_frames_db = np.percentile(energy_db, 15)

    return noise_frames_db


def spectral_subtraction(signal, sr, n_fft=128, hop_length=global_variables.HOP):
    """
    Perform spectral subtraction by estimating noise from the last 3 seconds of the audio.

    Parameters:
    - signal: Input audio signal (numpy array).
    - sr: Sampling rate of the audio signal.
    - n_fft: Number of FFT points.
    - hop_length: Hop length for FFT.

    Returns:
    - Denoised audio signal (numpy array).
    """
    noise_samples = 1 * sr

    noise_segment = signal[-noise_samples:]
    noise_stft = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length)
    noise_estimation = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    # Perform STFT on the original signal
    signal_stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # Spectral subtraction
    subtracted_magnitude = np.maximum(np.abs(signal_stft) - noise_estimation, 0)
    phase = np.angle(signal_stft)
    denoised_stft = subtracted_magnitude * np.exp(1j * phase)

    # Inverse STFT to convert back to time domain
    denoised_signal = librosa.istft(denoised_stft, hop_length=hop_length)

    return denoised_signal


def noise_gate(samples, sample_rate, threshold_dB=global_variables.NOISE_THRESHOLD):
    """
    Applies a noise gate to an audio array.

    Parameters:
    - samples: np.array, audio samples.
    - sample_rate: int, sample rate of the audio.
    - threshold_dB: float, the dB threshold below which the sound is gated.
    - fade_out_duration: int, duration in milliseconds to fade out to -110 dB then to silence.

    Returns:
    - np.array, processed audio samples.
    """
    # Calculate the threshold in linear scale
    threshold_amplitude = 10 ** (threshold_dB / 20)

    # Compute the envelope using RMS
    window_length = int(sample_rate * 0.1)  # window length of 100 ms
    envelope = np.array([
        np.sqrt(np.mean(samples[i:i + window_length] ** 2))
        for i in range(0, len(samples), window_length)
    ])

    # Determine where the envelope is below the threshold
    quiet = envelope < threshold_amplitude
    quiet = np.repeat(quiet, window_length)[:len(samples)]

    # Apply gating (silence the quiet parts)
    samples[quiet] = 0

    # Apply fade out at threshold crossing points
    for i in range(1, len(envelope)):
        if not quiet[i * window_length] and quiet[(i - 1) * window_length]:
            start = max(0, i * window_length - 1)
            end = min(len(samples), i * window_length)
            fade_samples = end - start
            if fade_samples > 0:
                fade_curve = np.linspace(1, 0, fade_samples)
                samples[start:end] *= fade_curve

    return samples


def noise_ceiling(samples, sample_rate, ceiling_dB=-(global_variables.TARGET_VOLUME * 30)):
    """
    Applies a noise ceiling to an audio array, silencing parts that exceed a loudness threshold.

    Parameters:
    - samples: np.array, audio samples.
    - sample_rate: int, sample rate of the audio.
    - ceiling_dB: float, the dB ceiling above which the sound is silenced.
    - fade_out_duration: int, duration in milliseconds to fade out the signal from the ceiling threshold to silence.

    Returns:
    - np.array, processed audio samples.
    """
    # Calculate the ceiling in linear scale
    ceiling_amplitude = 10 ** (ceiling_dB / 20)

    # Compute the envelope using RMS
    window_length = int(sample_rate * 0.1)  # window length of 100 ms
    envelope = np.array([
        np.sqrt(np.mean(samples[i:i + window_length] ** 2))
        for i in range(0, len(samples), window_length)
    ])

    # Determine where the envelope exceeds the ceiling
    loud = envelope > ceiling_amplitude
    loud = np.repeat(loud, window_length)[:len(samples)]

    # Apply gating (silence the loud parts)
    samples[loud] = 0

    # Apply fade out at threshold crossing points
    for i in range(1, len(envelope)):
        if loud[i * window_length] and not loud[(i - 1) * window_length]:
            start = max(0, i * window_length - 1)
            end = min(len(samples), i * window_length)
            fade_samples = end - start
            if fade_samples > 0:
                fade_curve = np.linspace(1, 0, fade_samples)
                samples[start:end] *= fade_curve

    return samples


def half_speed_keep_pitch(audio, frame_fft=2024, hop_len=global_variables.HOP):
    D = librosa.stft(audio, n_fft=frame_fft, hop_length=hop_len)
    audio_stretched = librosa.phase_vocoder(D, rate=0.5, hop_length=hop_len)
    audio_stretched = librosa.istft(audio_stretched, hop_length=hop_len)
    desired_length = 2 * len(audio)
    audio_stretched = librosa.util.fix_length(audio_stretched, size=desired_length)

    return audio_stretched


def plot_volume_over_time(audio, fs, hop_length=global_variables.HOP, frame_length=2048, label=None):
    title = 'Volume over Time '
    if label is not None:
        title += label

    envelope = np.abs(audio)
    envelope = librosa.util.frame(envelope, frame_length=frame_length, hop_length=hop_length).mean(axis=0)

    times = librosa.frames_to_time(np.arange(len(envelope)), sr=fs, hop_length=hop_length, n_fft=frame_length)

    plt.figure(figsize=(10, 4))
    plt.plot(times, envelope)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (Amplitude)')
    plt.show()


def plot_volume_over_time_db(audio, fs, hop_length=global_variables.HOP, frame_length=2048):
    amplitude = np.abs(audio)
    energy = amplitude ** 2
    energy = librosa.util.frame(energy, frame_length=frame_length, hop_length=hop_length).mean(axis=0)

    ref_energy = np.max(energy)
    energy_db = 10 * np.log10(np.maximum(energy, 1e-10) / ref_energy)  # Avoid log(0) by setting a floor at 1e-10

    times = librosa.frames_to_time(np.arange(len(energy_db)), sr=fs, hop_length=hop_length)

    # Plotting the volume over time in dB
    plt.figure(figsize=(10, 4))
    plt.plot(times, energy_db)
    plt.title('Volume over Time (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (dB)')
    plt.show()


def normalize_audio_over_time(audio: np.ndarray, resolution: int, target: float = global_variables.TARGET_VOLUME) -> np.ndarray:
    """
    Normalize audio parts in signal
    :param audio: ndarray of audio signal
    :param resolution: step for audio normalization
    :param target: target volume, default {global_variables.TARGET_VOLUME}
    :return: processed audio signal
    """
    # avoid /0 division
    epsilon = 1e-8
    for i in range(1, resolution + 1):
        audio_part = audio[int(len(audio) / resolution * (i - 1)):int(len(audio) / resolution * i)]
        factor = target / (audio_part.max() + epsilon)
        audio_part *= factor
        audio[int(len(audio) / resolution * (i - 1)):int(len(audio) / resolution * i)] = audio_part
    return audio


def play(audio, fs) -> None:
    """
    Plays audio
    :param audio: ndarray
    :param fs: float
    :return: None
    """
    sd.play(audio, fs)
    sd.wait()