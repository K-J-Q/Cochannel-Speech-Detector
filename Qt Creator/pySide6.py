import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile, QObject, QThread, Signal
from ui_livePred import Ui_MainWindow
import time, random

from pyqtgraph import BarGraphItem, ImageItem

# from torchaudio.io import StreamReader
# import torchaudio
# import torch

sys.path.append('./')
import machineLearning, testModel

class Worker(QThread):
    updatePrediction = Signal(list)

    def __init__(self):
        super(Worker, self).__init__()

    def run(self):
        print("Thread start")
        while not self.isInterruptionRequested():
            self.updatePrediction.emit([random.random(), random.random(), random.random()])
            time.sleep(1)
        # stream_iterator = streamer.stream()
        # while not self.isInterruptionRequested():
        #     (chunk,) = next(stream_iterator)

        #     wav, sr = augmentor.audio_preprocessing([wav, 8000])
        #     wav = torchaudio.functional.dcshift(wav, -wav.mean())
        #     wav = wav/wav.abs().max()
        #     pred = model(torch.unsqueeze(wav, dim=0).to(device))
        #     pred = sm(pred['out']) if isinstance(pred, dict) else sm(pred)
        #     self.updatePrediction.emit(pred)
        print("Thread complete")


class MainWindow(QMainWindow):
    specHistory = torch
    predHistory = []
    historyLength= 10

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.startStopButton.clicked.connect(self.buttonPressed)
        self.ui.mic_selector.currentIndexChanged.connect(self.getMicrophone)
        self.updateMicrophones()
        self.ui.graphWidget.setBackground('w')
        self.ui.graphWidget.setYRange(0, 2.5, padding=0)
        self.ui.graphWidget.setXRange(0, self.historyLength, padding=0)
        self.ui.graphWidget.getAxis('bottom').setTicks([[(i, str(i)) for i in range(self.historyLength + 1)]])
        self.ui.graphWidget.getAxis('left').setTicks([[(i, str(i)) for i in range(3)]])

        # make it a bar graph
        self.ui.graphWidget.addItem(BarGraphItem(x=[0], height=[0], width=1, brush='r'))

    def updateMicrophones(self):
        self.ui.mic_selector.clear()
        micName, self.micID = testModel.selectMicrophone(returnType='all')
        self.ui.mic_selector.addItem("Select microphone")
        self.ui.mic_selector.addItems(micName)

    def getMicrophone(self):
        mic = self.micID[self.ui.mic_selector.currentIndex() - 1]
        # streamer = StreamReader(
        #     src="audio=" + mic,
        #     format="dshow",
        #     buffer_size=8000 * windowLength * 2,
        # )
        # sm = torch.nn.Softmax(dim=1)
        # streamer.add_basic_audio_stream(
        #     frames_per_chunk=int(8000 * windowLength), sample_rate=8000)
    
    def updatePrediction(self, predictions):
        argmax = predictions.index(max(predictions))
        self.ui.class0.setText(f"{predictions[0]:.2f}")
        self.ui.class1.setText(f"{predictions[1]:.2f}")
        self.ui.class2.setText(f"{predictions[2]:.2f}")        
        self.predHistory.append(argmax)
        self.ui.model_output.setText(f"{argmax}")
        self.updateGraph()

    def updateGraph(self):
        x = list(range(len(self.predHistory), 0, -1))
        if len(self.predHistory) > self.historyLength:
            self.predHistory = self.predHistory[-self.historyLength:]
            x = x[-self.historyLength:]
        self.ui.graphWidget.clear()
        self.ui.graphWidget.addItem(BarGraphItem(x=x, height=self.predHistory, width=1, brush='r', pen='r'))
        self.ui.spectrogramGraphWidget.clear()

    def buttonPressed(self):
        if self.ui.startStopButton.text() == "Start":
            self.ui.startStopButton.setText("Stop")
            self.worker = Worker()
            self.worker.updatePrediction.connect(self.updatePrediction)
            self.worker.start()

        else:
            self.ui.startStopButton.setText("Start")
            self.worker.requestInterruption()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
