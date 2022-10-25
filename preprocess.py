import os
from tqdm import tqdm
import torch
import glob
import pathlib
import torchaudio
from Augmentation import Augmentor, getAudio

seconds = 4

if __name__ == "__main__":
    discarded = 0

    audioPaths = list(pathlib.Path(
        './data').glob('**/*.wav'))

    print(f"Total audio length {len(audioPaths)}")

    for audioIndex, audioPath in tqdm(enumerate(audioPaths), unit='files'):
        _, audioName = os.path.split(audioPath)
        aud = torchaudio.load(audioPath)
        augmentor = Augmentor()
        aud = augmentor.resample((augmentor.rechannel(aud)))

        #  trim audio
        x = torch.squeeze(aud[0])
        sampleLength = seconds * aud[1]

        for splitIndex, j in enumerate(range(0, len(x), sampleLength)):
            trimmed_audio = x[j:j+sampleLength]
            # print(splitIndex, ' ', len(trimmed_audio))
            if len(trimmed_audio) > sampleLength/2:
                torchaudio.save(
                    f'./data/{audioIndex}_{splitIndex}.wav', torch.unsqueeze(trimmed_audio, 0), aud[1])
            else:
                discarded += 1

    print(f'Total discarded length: {discarded}')
