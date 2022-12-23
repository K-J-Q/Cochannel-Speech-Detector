import argparse
import os
import pathlib
import subprocess

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import loader.utils as utils
from loader.AudioDataset import Augmentor


def detect_silence(path, threshold, duration=0.5):
    '''
    This function is a python wrapper to run the ffmpeg command in python and extranct the desired output

    path= Audio file path
    time = silence time threshold

    returns = list of tuples with start and end point of silences
    '''

    command = ["ffmpeg", "-i", path,
               "-af", f"silencedetect=n={threshold}:d={duration}", "-f", "null", "-"]
    out = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    s = stdout.decode("utf-8")

    k = s.split('[silencedetect @')
    if len(k) == 1:
        # print(stderr)
        return None

    start, end = [], []
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if i % 2 == 0:
            x = x.split('|')[0]
            x = x.split(':')[1].strip()
            end.append(float(x))
        else:
            x = x.split(':')[1]
            x = x.split('size')[0]
            x = x.replace('\r', '')
            x = x.replace('\n', '').strip()
            start.append(float(x))
    return list(zip(start, end))


def split_silences(audio, silence_list):
    aud, sr = audio
    aud = torch.squeeze(aud)
    mask = np.zeros([len(aud)], dtype=bool)

    for silence in silence_list:
        mask[int(silence[0] * sr):int(silence[1] * sr)] = True

    noise_aud = torch.unsqueeze(aud[mask], 0)
    speech_aud = torch.unsqueeze(aud[np.bitwise_not(mask)], 0)
    return (noise_aud, sr), (speech_aud, sr)


def main(input_path=None, output_path=None, mode=None):
    if input_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-path', type=str, required=True,
                            help='Path to the input audio files.')
        parser.add_argument('--output-path', type=str, required=True,
                            help='Path to save the processed audio files.')
        parser.add_argument('--mode', type=str, default='split', choices=['split', 'process'],
                            help='Whether to split the audio files into noise and speech or only process them.')
        args = parser.parse_args()
        input_path = args.input_path
        output_path = args.output_path
        mode = args.mode

    discarded = 0
    audioPaths = list(pathlib.Path(input_path).glob('**/*.wav'))

    augmentor = Augmentor()
    for audioIndex, audioPath in tqdm(enumerate(audioPaths), unit='files', total=len(audioPaths)):
        # only used if cut-off halfway

        if audioIndex >= 0:
            _, audioName = os.path.split(audioPath)
            aud = torchaudio.load(audioPath)
            # aud = augmentor.resample(augmentor.rechannel(aud), False)
            print(torch.mean(aud[0]))
            if mode == 'split':
                threshold = torch.median(aud[0][aud[0] > 0])

                silence_cutoff = threshold * 5
                speech_cutoff = threshold * 10

                silence_time = detect_silence(audioPath, silence_cutoff)
                speech_time = detect_silence(
                    audioPath, speech_cutoff if speech_cutoff > 0.03 else 0.05)

                aud_noise, _ = split_silences(aud, silence_time)
                _, aud_speech = split_silences(aud, speech_time)
                torchaudio.save(utils.uniquify(f'{output_path}/ENV/{audioName}'),
                                aud_noise[0], aud_noise[1])
                torchaudio.save(utils.uniquify(f'{output_path}/SPEECH/{audioName}'),
                                aud_speech[0], aud_speech[1])
            if mode == 'process':
                torchaudio.save(utils.uniquify(f'{output_path}/{audioName}'), aud[0], aud[1])


if __name__ == '__main__':
    main(input_path='C:/Users/Jian Quan/Desktop/0speaker_1.wav', output_path='E:/Processed Audio/backgroundNoise', mode='review')
