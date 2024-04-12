import matplotlib.pyplot as plt
import numpy as np
import librosa


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


def segment_normalization(audio, fs, segment_length=1.0, target_peak=0.15):
    """
    Normalize audio in segments to achieve a more uniform volume level across its duration.

    Parameters:
    audio (numpy array): Input audio signal.
    fs (int): Sampling rate of the audio signal.
    segment_length (float): Length of each segment to normalize, in seconds.
    target_peak (float): Target peak level for normalization.

    Returns:
    numpy array: The audio signal with normalized volume across segments.
    """
    samples_per_segment = int(segment_length * fs)
    num_segments = int(np.ceil(len(audio) / samples_per_segment))
    normalized_audio = np.zeros_like(audio)

    for i in range(num_segments):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        segment = audio[start_idx:end_idx]
        max_peak = np.abs(segment).max()
        if max_peak == 0:
            continue  # Avoid division by zero for silent segments
        normalization_factor = target_peak / max_peak
        normalized_audio[start_idx:end_idx] = segment * normalization_factor

    return normalized_audio


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


def apply_noise_gate(audio, sr, threshold_dB=-40, decay_rate=0.01):
    """
    Apply a noise gate with exponential decay to an audio signal.

    Parameters:
    - audio: The input audio signal (numpy array).
    - sr: Sampling rate of the audio signal.
    - threshold_dB: The level (in dB) below which the signal starts to decay.
    - decay_rate: The rate of exponential decay applied to the signal below the threshold.

    Returns:
    - The audio signal with the noise gate and decay applied.
    """
    # Convert threshold from dB to linear scale
    threshold_linear = 10 ** (threshold_dB / 20)

    # Initialize the output audio with the first sample
    gated_audio = np.zeros_like(audio)

    # Iterate through audio samples
    for i, sample in enumerate(audio):
        # Check if the sample's amplitude is below the threshold
        if abs(sample) < threshold_linear:
            if i == 0:
                gated_audio[i] = sample * decay_rate
            else:
                # Apply exponential decay from the previous sample
                gated_audio[i] = gated_audio[i - 1] * decay_rate
        else:
            # If above the threshold, pass the sample through
            gated_audio[i] = sample

    return gated_audio


def noise_gate(audio_segment, threshold_dB=-20, fade_out_duration=50):
    """
    Applies a noise gate to an audio segment.

    Parameters:
    - audio_segment: pydub.AudioSegment to be processed.
    - threshold_dB: The dB threshold below which the sound is gated.
    - fade_out_duration: Duration in milliseconds to fade out to -110 dB and then to silence.

    Returns:
    - Processed pydub.AudioSegment.
    """
    # Convert threshold dB to amplitude
    threshold_amplitude = audio_segment.dBFS - threshold_dB

    # Get audio data as array
    samples = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate

    # Compute the envelope using RMS
    window_length = sample_rate // 10  # window length of 100 ms
    envelope = np.array([
        np.sqrt(np.mean(samples[i:i + window_length] ** 2))
        for i in range(0, len(samples), window_length)
    ])

    # Normalize the envelope
    envelope -= threshold_amplitude
    envelope[envelope > 0] = 0
    envelope = np.repeat(envelope, window_length)[:len(samples)]

    # Apply gating
    gated_samples = samples * (envelope <= 0)

    # Apply fade out at threshold crossing points
    for i in range(1, len(envelope)):
        if envelope[i] == 0 and envelope[i - 1] != 0:
            start = max(0, i * window_length - fade_out_duration)
            end = i * window_length
            fade_curve = np.linspace(1, 0, end - start)
            gated_samples[start:end] *= fade_curve

    # Create new audio segment from gated samples
    gated_audio = audio_segment._spawn(gated_samples)

    return gated_audio


def noise_gate2(samples, sample_rate, threshold_dB=-20, fade_out_duration=50):
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