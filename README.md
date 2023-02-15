# Cochannel Speech Detector using Machine Learning

This project aims to develop a cochannel speech detector using machine learning techniques. The goal is to accurately count the number of speakers talking. This model aims to be lightweight with 0.7M parameters. It is capable of identifying cochannel noise with **92 %** accuracy and three simultanious speakers with **80 %** accuracy. 

Disclaimer: this accuracy metric was from high-quality audio, with little to no background noise .

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for *testing*

### Prerequisites

- Python 3.x

### Installing

```
git clone https://github.com/K-J-Q/cochannel-speech-detector.git
cd cochannel-speech-detector/release
pip install -r requirements.txt
```

### Running
#### Live prediction using GUI
`python livePredict.py`

![GUI](https://user-images.githubusercontent.com/71882072/218946682-e8f47490-19c4-4ef6-a7c3-7dcd9d6cfa5c.png)


#### Predicting audio wav (in either file or folder) 
The following arguments can be passed to the script:
--path: Required. The path to the file or folder to predict.
--model: Optional. The type of model to use for prediction (either 'speech' or 'radio'). Default is 'speech'.

Ground Truths (.txt) can be added in the same directory, and filename as wav file.

Cochannel speech detection
`python predict.py --path exampleAudio/trump.wav`

Cochannel radio detection
`python predict.py --path /path/to/file/or/folder --model radio`

![Picture1](https://user-images.githubusercontent.com/71882072/218946427-bedcf57b-6697-4cb8-a763-e37579ad615a.png)



## Dataset
National Speech Corpus was used for model development (https://www.imda.gov.sg/nationalspeechcorpus)
### Class 0: no speakers/environmental noise
https://user-images.githubusercontent.com/71882072/218943570-f8aa053e-53da-4bfb-9671-d82f58bb850a.mp4
### Class 1: one speaker
https://user-images.githubusercontent.com/71882072/218943714-ad34cd1d-d5e5-494e-ab42-ccf9e60dfb74.mp4
### Class 2: two simultaneous speakers
https://user-images.githubusercontent.com/71882072/218943726-6bfecf45-27cb-4ea7-8b2d-a56a59088966.mp4
### Class 3: three simultaneous speakers
https://user-images.githubusercontent.com/71882072/218943737-117a9080-1ef3-4ee4-a9b0-a1b1ad7d2d9a.mp4

## Built With

* [PyTorch](https://pytorch.org/) - The machine learning framework used
* [sounddevice](https://pypi.org/project/sounddevice/) - Interfacing with audio devices

## Authors

* **JQ** - *Initial work* -
