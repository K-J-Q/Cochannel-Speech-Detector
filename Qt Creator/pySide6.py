import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout,QLabel
from PySide6.QtCore import QFile, QObject, QThread, Signal, Qt
from ui_livePred import Ui_MainWindow
import time, random

from pyqtgraph import BarGraphItem, ImageItem, QtGui

# from torchaudio.io import StreamReader
import torchaudio
import torch

sys.path.append('./')
import machineLearning, testModel
from loader.AudioDataset import Augmentor

class Worker(QThread):
    updatePrediction = Signal(torch.Tensor, torch.Tensor)
    augmentor = Augmentor()
    sm = torch.nn.Softmax(dim=1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model):
        super(Worker, self).__init__()
        self.model = model.to(self.device)

    def run(self):
        print("Thread start")
        # while not self.isInterruptionRequested():
        #     pred = torch.rand([3])
        #     spec = torch.rand([100, 23])
        #     self.updatePrediction.emit(pred, spec)
        #     time.sleep(1)
        # stream_iterator = streamer.stream()
        while not self.isInterruptionRequested():
        #     (chunk,) = next(stream_iterator)
        #     wav = torch.from_numpy(chunk).float()
            wav = torch.rand([1, 8000])
            wav, sr = self.augmentor.audio_preprocessing([wav, 8000])
            wav = torchaudio.functional.dcshift(wav, -wav.mean())
            wav = wav/wav.abs().max()
            pred, spec = self.model(torch.unsqueeze(wav, dim=0).to(self.device))
            time.sleep(1)

            pred = self.sm(pred)
            pred = torch.round(pred[0])
            
            spec = spec.reshape(spec.shape[-2:])
            
            self.updatePrediction.emit(pred, spec)
        print("Thread complete")


class PredWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 80px;")
        layout.addWidget(self.label)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    historyLength= 10

    class SpectrogramShape:
        def __init__(self, width:int, height:int):
            self.width = width
            self.height = height
            

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # connect buttons
        self.ui.startStopButton.clicked.connect(self.buttonPressed)
        self.ui.mic_selector.currentIndexChanged.connect(self.getMicrophone)
        self.ui.refreshButton.clicked.connect(self.updateMicrophones)
        self.ui.onTopButton.clicked.connect(self.onTop)

        self.updateMicrophones()

        # Load model
        model, _, _ = machineLearning.selectTrainedModel()
        self.model = model.eval()

        # Prediction Graph
        self.predHistory = []
        self.labelGraph = self.ui.labelGraphWidget
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))

        self.labelGraph.setBackground(brush)
        self.labelGraph.setYRange(0, 2.5, padding=0)
        self.labelGraph.setXRange(0, self.historyLength, padding=0)
        self.labelGraph.getAxis('bottom').setTicks([[(i, str(i)) for i in range(self.historyLength + 1)]])
        self.labelGraph.getAxis('left').setTicks([[(i, str(i)) for i in range(3)]])
        
        # Spectrogram Graph
        spectrogramShape = self.model(torch.rand([1, 1, 8000]))[1].shape[-2:]
        self.specShape = self.SpectrogramShape(height=spectrogramShape[0], width=spectrogramShape[1])
        self.specHistory = torch.zeros((self.specShape.height, self.specShape.width*10))

        self.specGraph = self.ui.spectrogramGraphWidget

        self.specGraph.setBackground(brush)
        self.specGraph.showAxes(False)
        self.specGraph.setXRange(0, self.specShape.width*10, padding=0)
        self.specGraph.setYRange(0, self.specShape.height, padding=0)
        
        self.predWindow = PredWindow()

    def updateMicrophones(self):
        micName, self.micID = testModel.selectMicrophone(returnType='all')
        self.ui.mic_selector.clear()
        self.ui.mic_selector.addItem("Select microphone")
        self.ui.mic_selector.addItems(micName)

    def getMicrophone(self):
        mic = self.micID[self.ui.mic_selector.currentIndex() - 1]
        # streamer = StreamReader(
        #     src="audio=" + mic,
        #     format="dshow",
        #     buffer_size=8000 * windowLength * 2
        # )
        # streamer.add_basic_audio_stream(
        #     frames_per_chunk=int(8000 * windowLength), sample_rate=8000)
    
    def updatePrediction(self, predictions, spectrogram):
        argmax = predictions.argmax().item()
        self.ui.class0.setText(f"{predictions[0]}")
        self.ui.class1.setText(f"{predictions[1]}")
        self.ui.class2.setText(f"{predictions[2]}")        
        self.predHistory.append(argmax)
        self.appendSpectralData(spectrogram)
        self.ui.model_output.setText(f"{argmax}")
        self.predWindow.label.setText(f"{argmax}")

        self.updateGraph()

    def onTop(self):              
        self.predWindow.setWindowFlags(self.predWindow.windowFlags() | Qt.WindowStaysOnTopHint)
        self.predWindow.show()

    def appendSpectralData(self, spec):
        self.specHistory = torch.cat((spec, self.specHistory), dim=1)
        self.specHistory = self.specHistory[:, :-self.specShape.width]

    def updateGraph(self):
        x = list(range(len(self.predHistory), 0, -1))
        if len(self.predHistory) > self.historyLength:
            self.predHistory = self.predHistory[-self.historyLength:]
            x = x[-self.historyLength:]

        self.labelGraph.clear()
        self.labelGraph.addItem(BarGraphItem(
            x=x, height=self.predHistory, width=1, brush='r', pen='r'))

        
        spec = ImageItem(self.specHistory.T.numpy())
        spec.setColorMap('viridis')
        self.specGraph.clear()
        self.specGraph.addItem(spec)


    def buttonPressed(self):
        if self.ui.startStopButton.text() == "Start":
            self.ui.startStopButton.setText("Stop")
            self.worker = Worker(self.model)
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
