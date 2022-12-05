import sys
from PyQt6 import uic
from PyQt6 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("livePred.ui", self)
        self.modelOutputLabel = self.findChild(
            QtWidgets.QLabel, 'model_output')

    def updatePrediction(self, prediction: str):
        assert len(prediction) == 1, 'Only accepts strings of length 1'
        self.modelOutputLabel.setProperty('text', prediction)
        self.show()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
