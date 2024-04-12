import librosa
import numpy as np
import sounddevice as sd
import global_variables
import codes
import filters
import utilities
import global_variables

audio_path = r'./data/'


def get_non_silent_chunks(audio: np.ndarray) -> list:
    """
    function splits audio on silent parts to make it possible to analize code
    :param audio: he input audio signal.
    :return: list of audio chunks
    """
    non_silent_intervals = librosa.effects.split(audio, top_db=-global_variables.NOISE_THRESHOLD - 1, hop_length=32, frame_length=128)
    audio_segments = []
    for start_idx, end_idx in non_silent_intervals:
        segment = audio[start_idx:end_idx]
        audio_segments.append(segment)
    return audio_segments


def main(file: str) -> None:
    """
    main function, prints DTMF code from audio file
    :param file: name of audio file located in ./data/
    :return:
    """
    audio, fs = librosa.load(audio_path + file)
    audio = filters.select_freq(audio, fs)

    while utilities.estimate_noise_level(audio) > -55:
        audio = utilities.spectral_subtraction(audio, fs, 512, 128) #512 128 range(2)


    #audio = utilities.apply_noise_gate(audio, fs, threshold_dB=-48, decay_rate=0.0001)
    audio = utilities.noise_gate(audio, fs)
    # sd.play(audio, fs)
    # sd.wait()
    #

    audio_chunks = get_non_silent_chunks(audio)
    code = ''
    for chunk in audio_chunks:
        code += codes.extract_number(chunk, fs)
    print(code)
    # print(len(code))

    #


if __name__ == "__main__":
    main(r'challenge 2022.wav')

