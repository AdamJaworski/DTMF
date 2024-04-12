import matplotlib.pyplot as plt
import numpy as np
import librosa
import global_variables


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


def normalize_audio(audio: np.ndarray, target_peak: float = 0.1) -> np.ndarray:
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

    # Assume the lower 10th percentile of energy frames are noise
    noise_frames_db = np.percentile(energy_db, 15)

    return noise_frames_db


def spectral_subtraction(signal, sr, n_fft=2048, hop_length=512):
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
    # Calculate the number of samples corresponding to last 3 seconds
    noise_samples = 3 * sr

    # Ensure the signal is long enough for this operation
    if len(signal) < noise_samples:
        raise ValueError("Signal is shorter than 3 seconds.")

    # Use the last 3 seconds of the signal for noise estimation
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


def noise_gate(samples, sample_rate, threshold_dB=global_variables.NOISE_THRESHOLD, fade_out_duration=100):
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
            start = max(0, i * window_length - fade_out_duration)
            end = min(len(samples), i * window_length)
            fade_samples = end - start
            if fade_samples > 0:
                fade_curve = np.linspace(1, 0, fade_samples)
                samples[start:end] *= fade_curve

    return samples