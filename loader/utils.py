import os
from pathlib import Path
import shutil


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


def getAudioPaths(main_path, percent=0.9, repeatENVMul=0, repeatSPEECHMul=0):
    env_paths = list(Path(main_path+'/ENV').glob('**/*.wav'))
    speech_paths = list(Path(main_path+'/SPEECH').glob('**/*.wav'))

    train_env_size = int(len(env_paths) * percent)
    train_speech_size = int(len(speech_paths) * percent)

    for i in range(repeatENVMul):
        env_paths += env_paths
    for i in range(repeatSPEECHMul):
        speech_paths += speech_paths

    return ((env_paths[:train_env_size], speech_paths[:train_speech_size]), (env_paths[train_env_size:], speech_paths[train_speech_size:]))


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

import os
import importlib, inspect


def select_model():
    model_class = None
    # Get the list of files in the 'model' directory
    files = os.listdir('model')
    # Filter out any files that are not Python files
    files = [file for file in files if file.endswith('.py')]

    # Print a list of available models
    print('Available models:')
    for i, file in enumerate(files):
        print(f'{i+1}. {file[:-3]}')

    # Prompt the user to select a model
    while model_class is None:
        # Get the user's input
        selection = input('Enter the number of the model you want to use: ')
        try:
            # Convert the input to an integer
            selection = int(selection)
            # Check if the selection is within the range of available models
            if selection > 0 and selection <= len(files):
                # Import the selected module
                module = importlib.import_module(
                    'model.' + files[selection - 1][:-3])
                # Loop through the attributes in the module
                for attr in dir(module):
                    # Check if the attribute is a class
                    if inspect.isclass(getattr(module, attr)):
                        # Get the class
                        model_class = getattr(module, attr)
                        break
            else:
                # If the selection is not within the range of available models, print an error message
                print('Invalid selection')
        except ValueError:
            # If the input cannot be converted to an integer, print an error message
            print('Invalid input')

    return model_class

if __name__ == "__main__":
    model = select_model()
    print(model(512))
