import os
from tqdm import tqdm
import torch
import glob
import pathlib
import torchaudio
from Augmentation import Augmentor
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

seconds = 4

pathIN = filedialog.askdirectory(title='Select folder to process')
pathOUT = filedialog.askdirectory(
    title='Select folder to save the processed files')


# pathIN = 'E:/VocalSound'
# pathOUT = 'E:/Processed Audio/VocalSound'

# NOTE: Saving of multiple channels not yet implemented. May result in data wastage.
# Will indicate if rechanneled

# TODO: need to add initial checking of audio (maybe amplitude)


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
        # print(audioName)
        # print(aud[0].shape)
        aud = augmentor.resample(augmentor.rechannel(aud), False)

        #  trim audio
        x = torch.squeeze(aud[0])
        sampleLength = seconds * aud[1]

        for splitIndex, j in enumerate(range(0, len(x), sampleLength)):
            trimmed_audio = x[j:j+sampleLength]
            # print(splitIndex, ' ', len(trimmed_audio))
            # print(torch.sum(trimmed_audio > 0))
            if len(trimmed_audio) > sampleLength/2:
                torchaudio.save(
                    f'{pathOUT}/{audioName[0:-4]}_{splitIndex}.wav', torch.unsqueeze(trimmed_audio, 0), aud[1])
            else:
                discarded += 1

    print(f'Total discarded length: {discarded}')


if __name__ == "__main__":
    main()
    # cProfile.run(main())
