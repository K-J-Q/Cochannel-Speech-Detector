# Audio Processing Script

The code below is a Python script that processes a batch of audio files and splits them into two parts: speech and noise. The script uses the ffmpeg command to detect silences in the audio files and then uses the torchaudio library to split the audio files into speech and noise segments. The processed audio files can be saved to a specified output directory.

## Usage

To use the script, you need to provide the following arguments:

- `input_path`: The path to the directory containing the input audio files.
- `output_path`: The path to the directory where the processed audio files will be saved.
- `mode`: The processing mode to use. This can be either `split` to split the audio files into speech and noise segments, or `process` to only process the audio files without splitting them.

The script can be run either with or without command line arguments. To run the script with command line arguments, use a command like this:

```python
python script.py --input-path ./data --output-path E:/Processed Audio/train --mode split
```

To run the script without using command line arguments, you can call the `main` function directly and provide the values for the `input_path`, `output_path`, and `mode` arguments, like this:

```python
main(input_path='./data', output_path='E:/Processed Audio/train', mode='split')
```

The script uses the tqdm library to display a progress bar while processing the audio files. It also uses the Augmentor class from the loader. AudioDataset module to resample and rechannel the audio files before processing them.
