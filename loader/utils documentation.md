## getTransforms(augment)

This function returns a list of audio transformations that can be applied to audio files.

**Parameters**
- `augment` (bool): Whether to include the reverb transformation in the returned list.

**Returns**
- A list of audio transformations. If `augment` is `True`, the list will include a reverb transformation with a 70% wet signal. Otherwise, the list will be empty.

**Example**
```python
# Get a list of audio transformations with the reverb transformation
transforms = getTransforms(True)

# Get a list of audio transformations without the reverb transformation
transforms = getTransforms(False)
```

---

## uniquify(path, returnIndex=False)

This function takes in a file path and generates a unique filename by appending a counter to the end of the file if the original name already exists.

**Parameters**
- `path` (str): The original file path.
- `returnIndex` (bool, optional): Whether to also return the index of the generated filename. Default is `False`.

**Returns**
- The generated filename. If `returnIndex` is `True`, the function will also return the index of the generated filename.

**Example**
```python
# Generate a unique filename for the file at './audio/file.wav'
filename = uniquify('./audio/file.wav')

# Generate a unique filename for the file at './audio/file.wav' and get the index of the generated filename
filename, index = uniquify('./audio/file.wav', returnIndex=True)
```
---

## getAudioPaths(main_path, percent=0.9, repeatENVMul=0, repeatSPEECHMul=0)

This function returns the paths to the audio files in the `main_path` directory, split into train and test sets according to the `percent` parameter.

**Parameters**
- `main_path` (str): This is the root directory of the dataset. It should contain two subdirectories called "ENV" and "SPEECH", which in turn should contain the audio files.
- `percent` (float, optional): The percentage of the audio files to include in the train set. Default is 0.9.
- `repeatENVMul` (int, optional): The number of times to repeat the environmental audio paths. Default is 0.
- `repeatSPEECHMul` (int, optional): The number of times to repeat the speech audio paths. Default is 0.

**Returns**
- A tuple containing two tuples: the first tuple contains the paths to the environmental and speech audio files in the train set, and the second tuple contains the paths to the environmental and speech audio files in the test set.

**Example**
```python
# Split the dataset into training and validation sets
(train_env, train_speech), (val_env, val_speech) = getAudioPaths('/path/to/dataset', percent=0.8)

(train_env, train_speech), (val_env, val_speech) = getAudioPaths('/path/to/dataset', percent=0.8, repeatENVMul=5, repeatSPEECHMul=5)
```
---

## clearUselesslogs(minFiles=1)

This function deletes empty folders in the `./logs` directory. Folders with less than `minFiles` files will be considered empty.

**Parameters**
- `minFiles` (int, optional): The minimum number of files required for a folder to be considered non-empty. Default is 1.

**Returns**
- None

**Example**
```python
# Delete empty folders in the './logs' directory with less than 1 file
clearUselesslogs()

# Delete empty folders in the './logs' directory with less than 5 files
clearUselesslogs(minFiles=5)
```
---

## removeHparams()

This function deletes all subfolders in the `./logs/` directory.

**Parameters**
- None

**Returns**
- None

**Example**
```python
# Delete all subfolders in the './logs/' directory
removeHparams()
```
---

## select_model()

This function prompts the user to select a model from the `model` directory, imports the selected module, and returns the model class.

**Parameters**
- None

**Returns**
- The selected model class.

**Example**
```python
# Prompt the user to select a model from the './model' directory and import the selected module
model_class = select_model()

# Create an instance of the selected model
model = model_class()
```