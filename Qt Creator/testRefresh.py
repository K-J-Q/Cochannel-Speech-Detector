import sys
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
import time

class DataSource(QObject):
    data_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._data = 'Hello, World!'

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.data_changed.emit(value)


class MainWindow(QMainWindow):
    def __init__(self, data_source):
        super().__init__()
        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        # Connect the data_source's data_changed signal to the refresh slot
        data_source.data_changed.connect(self.refresh)

    @pyqtSlot(str)
    def refresh(self, data):
        # Update the text of the label
        self.label.setText(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_source = DataSource()
    window = MainWindow(data_source)
    window.show()
    for i in range(10):
        data_source.data = str(i)
        app.processEvents()
        time.sleep(1)
    sys.exit(app.exec_())
    