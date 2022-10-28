import os
from tqdm import tqdm
import torch
import glob
import pathlib
import torchaudio
from Augmentation import Augmentor

seconds = 4
pathIN = 'C:/Users/Jian Quan/Desktop/'
pathOUT = 'E:/Singapore Speech Corpus/WAVE'

# NOTE: Saving of multiple channels not yet implemented. May result in data wastage

if __name__ == "__main__":
    discarded = 0

    audioPaths = list(pathlib.Path(pathIN).glob('**/*.wav'))

    print(f"Total audio length {len(audioPaths)}")
    augmentor = Augmentor()
    for audioIndex, audioPath in tqdm(enumerate(audioPaths), unit='files'):
        _, audioName = os.path.split(audioPath)
        aud = augmentor.resample(
            augmentor.rechannel(torchaudio.load(audioPath)))

        #  trim audio
        x = torch.squeeze(aud[0])
        sampleLength = seconds * aud[1]

        for splitIndex, j in enumerate(range(0, len(x), sampleLength)):
            trimmed_audio = x[j:j+sampleLength]
            # print(splitIndex, ' ', len(trimmed_audio))
            if len(trimmed_audio) > sampleLength/2:
                torchaudio.save(
                    f'{pathOUT}{audioIndex}_{splitIndex}.wav', torch.unsqueeze(trimmed_audio, 0), aud[1])
            else:
                discarded += 1

    print(f'Total discarded length: {discarded}')
