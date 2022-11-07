import os
from pathlib import Path
import audiomentations
import torchaudio


def getTransforms(augment):
    if augment:
        return [
            {
                # "before_cochannel": [
                # audiomentations.Gain(
                #     min_gain_in_db=-3, max_gain_in_db=3, p=0.5,),
                # audiomentations.TimeStretch(min_rate=0.8,
                #                             max_rate=1.2,
                #                             p=0.5,
                #                             leave_length_unchanged=False,),
                # ],
                # "audio": [
                #     audiomentations.AddGaussianNoise(min_amplitude=0.001,
                #                                      max_amplitude=0.025,
                #                                      p=0.5),
                # ],
                "spectrogram": [
                    torchaudio.transforms.TimeMasking(80),
                    torchaudio.transforms.FrequencyMasking(80)
                ],
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


# TODO: output train and val set instead of random split
def getAudioPaths(main_path, percent = 0.9, repeatENVMul=0, repeatSPEECHMul = 0):
    env_paths = list(Path(main_path+'/ENV').glob('**/*.wav'))
    speech_paths = list(Path(main_path+'/SPEECH').glob('**/*.wav'))

    train_env_size = int(len(env_paths)* percent)
    train_speech_size = int(len(speech_paths)* percent)

    for i in range(repeatENVMul):
        env_paths += env_paths
    for i in range(repeatSPEECHMul):
        speech_paths += speech_paths

    return ((env_paths[:train_env_size], speech_paths[:train_speech_size]), (env_paths[train_env_size:], speech_paths[train_speech_size:]))


if __name__ == "__main__":
    print(getTransforms(True))
