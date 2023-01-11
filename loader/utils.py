import os
import sys
from pathlib import Path
import shutil
import importlib
import inspect


def getTransforms(augment):
    if augment:
        return [
            {
                "before_cochannel_sox": [["reverb", "70"], ['channels', '1']],
            },
        ]
    else:
        return [{}]


def uniquify(path, returnIndex=False):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename} ({str(counter)}){extension}"
        counter += 1
    if returnIndex:
        return path, counter - 1
    return path


def getAudioPaths(main_path, percent=0.9):
    env_paths = list(Path(main_path+'/ENV').glob('*.wav'))
    speech_paths = list(Path(main_path+'/SPEECH').glob('*.wav'))
    train_env_size = int(len(env_paths) * percent)
    train_speech_size = int(len(speech_paths) * percent)
    return (env_paths[:train_env_size], speech_paths[:train_speech_size]), (env_paths[train_env_size:], speech_paths[train_speech_size:])

def clearUselesslogs(minFiles=1):
    logs = list(Path('./logs').glob('*'))
    deletedCount = 0
    for log in logs:
        if log != Path('logs/traces') and len(list(log.glob('**/*.*'))) <= minFiles:
            shutil.rmtree(log)
            deletedCount += 1

    if deletedCount:
        print(f'{deletedCount} folders deleted')


def removeHparams():
    logFolders = list(Path('./logs/').glob('*/*'))
    deletedCount = 0
    for log in logFolders:
        if not Path.is_file(log):
            shutil.rmtree(log)
            deletedCount += 1

    if deletedCount:
        print(f'{deletedCount} folders deleted')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    pass
