import torch
from Augmentation import Augmentor

if __name__ == "__main__":

    # simulate 13.5 seconds audio clip
    aud = [[torch.rand(108000)], 12000]

    augmentor = Augmentor()
    aud = augmentor.resample((augmentor.rechannel(aud)))
    print(aud[1])

#  trim audio
    x = aud[0]
    sampleLength = 1 * aud[1]

    for i, j in enumerate(range(0, len(x), sampleLength)):
        trimmed_audio = x[j:j+sampleLength]
        print(i, ' ', len(trimmed_audio))
