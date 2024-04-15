import librosa
import numpy as np
import codes
import filters
import utilities
import global_variables
import difflib


audio_path = r'./data/'


def get_non_silent_chunks(audio: np.ndarray, fs) -> list:
    """
    function splits audio on silent parts to make it possible to analize code
    :param audio: he input audio signal.
    :return: list of audio chunks
    """
    non_silent_intervals = librosa.effects.split(audio, top_db=-global_variables.NOISE_THRESHOLD, hop_length=global_variables.HOP, frame_length=int(fs/8))
    audio_segments = []
    for start_idx, end_idx in non_silent_intervals:
        segment = audio[start_idx:end_idx]
        audio_segments.append(segment)
    return audio_segments


def main(file: str) -> None:
    """
    main function, prints DTMF code from audio file
    :param file: name of audio file located in ./data/
    :return: code as string
    """
    audio, fs = librosa.load(file)

    audio = filters.select_freq(audio, fs)     # selecting freq

    audio = utilities.normalize_audio_over_time(audio, resolution=int(len(audio) / fs))     # normalizing every second

    while utilities.estimate_noise_level(audio) > -38:   # global_variables.NOISE_THRESHOLD:    # Lowering ground
        audio = utilities.spectral_subtraction(audio, fs)

    audio = utilities.noise_ceiling(audio, fs)      # flatten top
    audio = utilities.noise_gate(audio, fs)         # remove noisy bottom

    audio_chunks = get_non_silent_chunks(audio, fs)

    # setting code
    codes_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '0', '#']
    for index in range(len(codes_list)):
        while not codes.set_keys(audio_chunks[index], fs, codes_list[index]):
            del audio_chunks[index]
    print(global_variables.new_keys)


    code = ''
    for chunk in audio_chunks[12:]:
        code += codes.get_code(chunk, fs)

    print(code)


def test_for_nn(file: str) -> None:
    audio, fs = librosa.load(file)
    utilities.display_freq(audio, fs)


if __name__ == "__main__":
    # nn_data = r"D:\DTMF_ML\Data\verified"
    # for file in os.listdir(nn_data):
    #     print(file)
    #     test_for_nn(f"{nn_data}\{file}")

    main(audio_path + r'challenge 2024.wav')
    # for i in range(10):
    #     main(rf's{i}.wav')
    # for i in range(12):


