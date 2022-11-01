import os

import audiomentations
import torchaudio


def getTransforms(augment):
    if augment:
        return [
            {
                "before_cochannel": [
                    audiomentations.Gain(
                        min_gain_in_db=0, max_gain_in_db=3, p=0.5,),
                    audiomentations.TimeStretch(min_rate=0.8,
                                                max_rate=1.2,
                                                p=0.5,
                                                leave_length_unchanged=False,),
                ],
                "audio": [
                    audiomentations.AddGaussianNoise(min_amplitude=0.001,
                                                     max_amplitude=0.025,
                                                     p=0.5),
                ],
                # "spectrogram": [
                #     torchaudio.transforms.TimeMasking(80),
                #     torchaudio.transforms.FrequencyMasking(80)
                # ],
            },
        ]
    else:
        return [{}]


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename} ({str(counter)}){extension}"
        counter += 1

    return path
if __name__ == "__main__":
    print(getTransforms(True))