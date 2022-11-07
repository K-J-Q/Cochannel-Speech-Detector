from scipy.io.wavfile import read, write
import subprocess
from scipy.io import wavfile
import numpy as np
import os
from tqdm import tqdm
import torch
import glob
import pathlib
import torchaudio
from loader.AudioDataset import Augmentor
import tkinter as tk
from tkinter import filedialog

# pathIN = 'E:/VocalSound'
# pathOUT = 'E:/Processed Audio/VocalSound'

# NOTE: Saving of multiple channels not yet implemented. May result in data wastage.
# Will indicate if rechanneled

root = tk.Tk()
root.withdraw()

pathIN = filedialog.askdirectory(title='Select folder to process')


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
        mask[int(silence[0]*sr):int(silence[1]*sr)] = True
    noise_aud = torch.unsqueeze(aud[mask], 0)
    speech_aud = torch.unsqueeze(aud[np.bitwise_not(mask)], 0)
    return (noise_aud, sr), (speech_aud, sr)


def main():
    discarded = 0

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    audioPaths = list(pathlib.Path(pathIN).glob('**/*.wav'))

    print(f"Total audio length {len(audioPaths)}")
    augmentor = Augmentor()

    for audioIndex, audioPath in tqdm(enumerate(audioPaths), unit='files'):
        _, audioName = os.path.split(audioPath)

        aud = torchaudio.load(audioPath)
        threshold = torch.median(aud[0][aud[0] > 0])*6

        silence_list = detect_silence(audioPath, threshold)
 
        aud = augmentor.resample(augmentor.rechannel(aud), False)
        aud_noise, aud_speech = split_silences(aud, silence_list)
        torchaudio.save(f'E:/Processed Audio/ENV/4 Diff Room/{audioName}.wav',
                        aud_noise[0], aud_noise[1])
        torchaudio.save(f'E:/Processed Audio/SPEECH/4 Diff Room/{audioName}.wav',
                        aud_speech[0], aud_speech[1])

    print(f'Total discarded length: {discarded}')


if __name__ == "__main__":
    main()
    # cProfile.run(main())
