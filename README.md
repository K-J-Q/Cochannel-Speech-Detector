# Cochannel Speech Detector using Machine Learning

This project aims to develop a cochannel speech detector using machine learning techniques. The goal is to accurately count the number of speakers talking.

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

#### Predicting audio wav (in either file or folder) 
The following arguments can be passed to the script:
--path: Required. The path to the file or folder to predict.
--model: Optional. The type of model to use for prediction (either 'speech' or 'radio'). Default is 'speech'.

Cochannel speech detection
`python filename.py --path /path/to/file/or/folder`

Cochannel radio detection
`python filename.py --path /path/to/file/or/folder --model radio`



## Built With

* [PyTorch](https://pytorch.org/) - The machine learning framework used
* [sounddevice](https://pypi.org/project/sounddevice/) - Interfacing with audio devices

## Authors

* **JQ** - *Initial work* -
