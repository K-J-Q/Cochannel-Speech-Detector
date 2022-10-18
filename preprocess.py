import torch
import torchaudio
from Augmentation import Augmentor, getAudio

if __name__ == "__main__":
    audioPaths = getAudio('./data/raw')
    for audioIndex, audioPath in enumerate(audioPaths):
        aud = torchaudio.load(audioPath)
        augmentor = Augmentor()
        aud = augmentor.resample((augmentor.rechannel(aud)))

    #  trim audio
        x = torch.squeeze(aud[0])
        sampleLength = 1 * aud[1]

        for splitIndex, j in enumerate(range(0, len(x), sampleLength)):
            trimmed_audio = x[j:j+sampleLength]
            # print(splitIndex, ' ', len(trimmed_audio))
            torchaudio.save(f'./data/processed/1_{audioIndex}_{splitIndex}.wav',
                            torch.unsqueeze(trimmed_audio, 0), aud[1])