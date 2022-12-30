import sys
from PyQt6 import uic
from PyQt6 import QtCore, QtGui, QtWidgets
import time
import sys
from torchaudio.io import StreamReader
import torchaudio
import torch


sys.path.append('./')
import machineLearning, testModel
from loader.AudioDataset import Augmentor

class MainWindow(QtWidgets.QMainWindow):
    micChanged = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("Qt Creator/livePred.ui", self)
        self.modelOutputLabel = self.findChild(
            QtWidgets.QLabel, 'model_output')
        self.class0ProbabilityLabel = self.findChild(
            QtWidgets.QLabel, 'class0')
        self.class1ProbabilityLabel = self.findChild(
            QtWidgets.QLabel, 'class1')
        self.class2ProbabilityLabel = self.findChild(
            QtWidgets.QLabel, 'class2')
        self.startStopButton = self.findChild(
            QtWidgets.QPushButton, 'startStopButton')
        self.startStopButton.clicked.connect(self.startStopButtonClicked)
        self.micSelector = self.findChild(QtWidgets.QComboBox, 'mic_selector')
        self.micSelector.currentIndexChanged.connect(self.micSelectorChanged)
        self.progressBar = self.findChild(QtWidgets.QProgressBar, 'progressBar')

    def updateMicSelector(self, mics):
        self.micSelector.clear()
        self.micSelector.addItems(mics)

    def updateModelOutput(self, probabilities):
        probabilities = [str(round(prob, 4)) for prob in probabilities]
        self.class0ProbabilityLabel.setText(probabilities[0])
        self.class1ProbabilityLabel.setText(probabilities[1])
        self.class2ProbabilityLabel.setText(probabilities[2])

        prediction = str(probabilities.index(max(probabilities)))
        self.modelOutputLabel.setText(prediction)
    
    def startStopButtonClicked(self):
        self.startStopButton.setText('Stop') if self.startStopButton.text() == 'Start' else self.startStopButton.setText('Start')

    def micSelectorChanged(self):
        self.micChanged = True

    def checkMicChanged(self):
        if self.micChanged:
            self.micChanged = False
            return True
        else:
            return False

model, device, epoch = machineLearning.selectTrainedModel(setCPU=True)
model = model.to(device)
model.eval()

windowLength = 1
augmentor = Augmentor()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()

# update microphone selector
micName, micID = testModel.selectMicrophone(returnType='all')
window.updateMicSelector(micName)

while True:
    if window.checkMicChanged():
        streamer = StreamReader(
            src="audio=" + micID[window.micSelector.currentIndex()],
            format="dshow",
            buffer_size=8000 * windowLength * 2,
        )

        sm = torch.nn.Softmax(dim=1)
        streamer.add_basic_audio_stream(
            frames_per_chunk=int(8000 * windowLength), sample_rate=8000)
        stream_iterator = streamer.stream()

    wav = torch.zeros(1, 8000 * windowLength)

    while True:
        (chunk,) = next(stream_iterator)
        
        wav, sr = augmentor.audio_preprocessing([wav, 8000])
        wav = torchaudio.functional.dcshift(wav, -wav.mean())
        wav = wav/wav.abs().max()
        pred = model(torch.unsqueeze(wav, dim=0).to(device))
        pred = sm(pred['out']) if isinstance(pred, dict) else sm(pred)
        window.updateModelOutput(pred[0].tolist())
        app.processEvents()

app.exec()
