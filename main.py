import time

import librosa
import numpy as np
import codes
import filters
import utilities
import global_variables
import difflib


audio_path = r'./data/'


def get_non_silent_chunks(audio: np.ndarray) -> list:
    """
    function splits audio on silent parts to make it possible to analize code
    :param audio: he input audio signal.
    :return: list of audio chunks
    """
    non_silent_intervals = librosa.effects.split(audio, top_db=-(global_variables.NOISE_THRESHOLD * 2) + 0, hop_length=16, frame_length=524)
    audio_segments = []
    for start_idx, end_idx in non_silent_intervals:
        segment = audio[start_idx:end_idx]
        audio_segments.append(segment)
    return audio_segments


def main(file: str, value_to_adjust=None) -> str:
    """
    main function, prints DTMF code from audio file
    :param file: name of audio file located in ./data/
    :return: code as string
    """
    audio, fs = librosa.load(audio_path + file)

    audio = filters.select_freq(audio, fs)

    audio = utilities.normalize_audio_over_time(audio, resolution=int(len(audio) / fs))

    for i in range(2):
        audio = utilities.spectral_subtraction(audio, fs, 128, 32)

    audio = utilities.noise_ceiling(audio, fs)

    audio = utilities.noise_gate(audio, fs)

    #utilities.plot_volume_over_time(audio, fs)
    audio_chunks = get_non_silent_chunks(audio)
    code = ''
    for chunk in audio_chunks:
        code += codes.extract_number(chunk, fs)
    return code


if __name__ == "__main__":
    output = main(r'challenge 2022.wav')
    expected_output = '123456789*0#874995*888285837**40#*9135*351#043387301#149951161#978567013353#'
    loss = len(list(difflib.ndiff(output, expected_output)))
    print(output)
    print(loss)
    # for i in range(10):
    #     main(rf's{i}.wav')

