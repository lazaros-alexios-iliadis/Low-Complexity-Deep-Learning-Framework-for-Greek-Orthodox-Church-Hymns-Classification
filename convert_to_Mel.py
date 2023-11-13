import librosa
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment

audio_path = "AudioDataset/"


def convert2mel(file_path, file_wav, n_fft, hop_length):
    # load the file using librosa with default sample rate for music processing of 22050
    signal, sr = librosa.load(file_wav, sr=22050)

    # short time Fourier transform
    mel = librosa.feature.melspectrogram(y=signal, hop_length=hop_length, n_fft=n_fft, n_mels=128)
    spectrogram = np.abs(mel)

    # print(spectrogram)

    # log spectrogram for better visualization
    log_spec = librosa.amplitude_to_db(spectrogram)

    # display spectrogram
    librosa.display.specshow(log_spec, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length, cmap='magma')
    plt.xlabel('Time', size=20)
    plt.ylabel('Frequency', size=20)

    # if needed
    # plt.colorbar()

    # replace last three elements of the string to save it
    replace_last_three_elements = ''
    file_path = replace_last_three_elements.join(file_path.rsplit(file_path[-4:], 1))

    # save the spectrograms to new file e.g. Plots after replacing some strings
    plt.savefig(f"{file_path.replace('AudioDataset/', 'Mel/')}_melspectrogram.jpg")

    # clear the plot to plot new spectrogram
    plt.clf()

    # function to find what type is the audio file


def convert_to_wav_if_needed(file_path):
    if file_path[-3:] == 'mp3':
        file_wav = convert_mp3_to_wav(file_path)
    elif file_path[-3:] == 'aac':
        file_wav = convert_aac_to_wav(file_path)
    elif file_path[-3:] == 'ogg':
        file_wav = convert_ogg_to_wav(file_path)
    elif file_path[-3:] == 'm4a':
        file_wav = convert_m4a_to_wav(file_path)
    else:
        file_wav = file_path
    return file_wav

    # functions to convert any kind of audio to WAV


def convert_mp3_to_wav(file_path):
    track = AudioSegment.from_file(file_path, format='mp3')

    file_wav = track.export(format='wav')

    return file_wav


def convert_m4a_to_wav(file_path):
    track = AudioSegment.from_file(file_path, format='m4a')

    file_wav = track.export(format='wav')

    return file_wav


def convert_ogg_to_wav(file_path):
    track = AudioSegment.from_file(file_path, format='ogg')

    file_wav = track.export(format='wav')

    return file_wav


def convert_aac_to_wav(file_path):
    track = AudioSegment.from_file(file_path, format='aac')

    file_wav = track.export(format='wav')

    return file_wav


# loop through the files
for dirpath, _, filenames in os.walk(audio_path):

    # start processing the files
    for i, f in enumerate(filenames):
        # full path for loading the file
        file_path = os.path.join(dirpath, f)

        # convert to wav file if needed
        file_wav = convert_to_wav_if_needed(file_path)

        # convert to spectrogram
        convert2mel(file_path, file_wav, n_fft=2048, hop_length=512)
