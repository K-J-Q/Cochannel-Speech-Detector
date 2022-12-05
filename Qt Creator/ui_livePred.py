# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'livePred.ui'
##
## Created by: Qt User Interface Compiler version 6.4.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSplitter,
    QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(639, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setGeometry(QRect(160, 70, 331, 391))
        self.splitter_2.setOrientation(Qt.Vertical)
        self.splitter = QSplitter(self.splitter_2)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.label = QLabel(self.splitter)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.label)
        self.comboBox = QComboBox(self.splitter)
        self.comboBox.setObjectName(u"comboBox")
        self.splitter.addWidget(self.comboBox)
        self.splitter_2.addWidget(self.splitter)
        self.model_output = QLabel(self.splitter_2)
        self.model_output.setObjectName(u"model_output")
        font1 = QFont()
        font1.setPointSize(150)
        self.model_output.setFont(font1)
        self.model_output.setAlignment(Qt.AlignCenter)
        self.splitter_2.addWidget(self.model_output)
        self.startStopButton = QPushButton(self.splitter_2)
        self.startStopButton.setObjectName(u"startStopButton")
        self.splitter_2.addWidget(self.startStopButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 639, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Microphone", None))
        self.model_output.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.startStopButton.setText(QCoreApplication.translate("MainWindow", u"Start", None))
    # retranslateUi

