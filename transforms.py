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
                    audiomentations.Shift(min_fraction=-1,
                                          max_fraction=1,
                                          p=0.5,)
                ],
                "audio": [
                    audiomentations.AddGaussianNoise(min_amplitude=0.001,
                                                     max_amplitude=0.025,
                                                     p=0.5),
                    audiomentations.PitchShift(min_semitones=-4,
                                               max_semitones=4,
                                               p=0.5),
                ],
            },
            {
                "before_cochannel": [
                    audiomentations.Gain(
                        min_gain_in_db=0, max_gain_in_db=3, p=0.5),
                    audiomentations.TimeStretch(min_rate=0.8,
                                                max_rate=1.2,
                                                p=0.5,
                                                leave_length_unchanged=False,),
                    audiomentations.Shift(min_fraction=-1,
                                          max_fraction=1,
                                          p=0.5,)
                ],
                "spectrogram": [
                    torchaudio.transforms.TimeMasking(80),
                    torchaudio.transforms.FrequencyMasking(80)
                ],
            },
        ]
    else:
        return []


if __name__ == "__main__":
    print(getTransforms(True))
