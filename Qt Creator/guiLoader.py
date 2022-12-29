import sys
from PyQt6 import uic
from PyQt6 import QtCore, QtGui, QtWidgets
import time
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("Qt Creator/livePred.ui", self)
        self.modelOutputLabel = self.findChild(
            QtWidgets.QLabel, 'model_output')
    def updateModelOutput(self, prediction: str):
        assert len(prediction) == 1, 'Only accepts strings of length 1'
        self.modelOutputLabel.setText(prediction)
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()

for i in range(10):
    window.updateModelOutput(str(i))
    app.processEvents()
    time.sleep(1)
app.exec()
