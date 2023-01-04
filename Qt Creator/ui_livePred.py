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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QSplitter, QStatusBar, QToolButton, QVBoxLayout,
    QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(960, 453)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setGeometry(QRect(30, 20, 321, 391))
        self.splitter.setOrientation(Qt.Vertical)
        self.mic_selection = QSplitter(self.splitter)
        self.mic_selection.setObjectName(u"mic_selection")
        self.mic_selection.setOrientation(Qt.Vertical)
        self.label = QLabel(self.mic_selection)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)
        self.mic_selection.addWidget(self.label)
        self.mic_selector = QComboBox(self.mic_selection)
        self.mic_selector.setObjectName(u"mic_selector")
        self.mic_selector.setEditable(True)
        self.mic_selection.addWidget(self.mic_selector)
        self.splitter.addWidget(self.mic_selection)
        self.model_output = QLabel(self.splitter)
        self.model_output.setObjectName(u"model_output")
        font1 = QFont()
        font1.setPointSize(150)
        self.model_output.setFont(font1)
        self.model_output.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.model_output)
        self.verticalLayoutWidget = QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.probability = QHBoxLayout()
        self.probability.setObjectName(u"probability")
        self.class0 = QLabel(self.verticalLayoutWidget)
        self.class0.setObjectName(u"class0")
        font2 = QFont()
        font2.setBold(False)
        self.class0.setFont(font2)
        self.class0.setAlignment(Qt.AlignCenter)

        self.probability.addWidget(self.class0)

        self.class1 = QLabel(self.verticalLayoutWidget)
        self.class1.setObjectName(u"class1")
        self.class1.setFont(font2)
        self.class1.setAlignment(Qt.AlignCenter)

        self.probability.addWidget(self.class1)

        self.class2 = QLabel(self.verticalLayoutWidget)
        self.class2.setObjectName(u"class2")
        self.class2.setFont(font2)
        self.class2.setAlignment(Qt.AlignCenter)

        self.probability.addWidget(self.class2)


        self.verticalLayout.addLayout(self.probability)

        self.probability_labels = QHBoxLayout()
        self.probability_labels.setObjectName(u"probability_labels")
        self.label_class0 = QLabel(self.verticalLayoutWidget)
        self.label_class0.setObjectName(u"label_class0")
        font3 = QFont()
        font3.setBold(True)
        self.label_class0.setFont(font3)
        self.label_class0.setAlignment(Qt.AlignCenter)

        self.probability_labels.addWidget(self.label_class0)

        self.label_class1 = QLabel(self.verticalLayoutWidget)
        self.label_class1.setObjectName(u"label_class1")
        self.label_class1.setFont(font3)
        self.label_class1.setAlignment(Qt.AlignCenter)

        self.probability_labels.addWidget(self.label_class1)

        self.label_class2 = QLabel(self.verticalLayoutWidget)
        self.label_class2.setObjectName(u"label_class2")
        self.label_class2.setFont(font3)
        self.label_class2.setAlignment(Qt.AlignCenter)

        self.probability_labels.addWidget(self.label_class2)


        self.verticalLayout.addLayout(self.probability_labels)

        self.splitter.addWidget(self.verticalLayoutWidget)
        self.startStopButton = QPushButton(self.splitter)
        self.startStopButton.setObjectName(u"startStopButton")
        self.splitter.addWidget(self.startStopButton)
        self.spectrogramGraphWidget = PlotWidget(self.centralwidget)
        self.spectrogramGraphWidget.setObjectName(u"spectrogramGraphWidget")
        self.spectrogramGraphWidget.setGeometry(QRect(390, 20, 541, 291))
        self.labelGraphWidget = PlotWidget(self.centralwidget)
        self.labelGraphWidget.setObjectName(u"labelGraphWidget")
        self.labelGraphWidget.setGeometry(QRect(390, 310, 541, 91))
        self.refreshButton = QToolButton(self.centralwidget)
        self.refreshButton.setObjectName(u"refreshButton")
        self.refreshButton.setGeometry(QRect(320, 25, 22, 22))
        icon = QIcon()
        icon.addFile(u"Qt Creator/resources/refreshIcon.png", QSize(), QIcon.Normal, QIcon.Off)
        self.refreshButton.setIcon(icon)
        self.onTopButton = QPushButton(self.centralwidget)
        self.onTopButton.setObjectName(u"onTopButton")
        self.onTopButton.setGeometry(QRect(280, 90, 31, 31))
        icon1 = QIcon()
        icon1.addFile(u"Qt Creator/resources/windowIcon.png",
                      QSize(), QIcon.Normal, QIcon.Off)
        self.onTopButton.setIcon(icon1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 960, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Input Source", None))
        self.model_output.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.class0.setText(QCoreApplication.translate("MainWindow", u"0.12", None))
        self.class1.setText(QCoreApplication.translate("MainWindow", u"0.80", None))
        self.class2.setText(QCoreApplication.translate("MainWindow", u"0.08", None))
        self.label_class0.setText(QCoreApplication.translate("MainWindow", u"Class 0", None))
        self.label_class1.setText(QCoreApplication.translate("MainWindow", u"Class 1", None))
        self.label_class2.setText(QCoreApplication.translate("MainWindow", u"Class 2", None))
        self.startStopButton.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.refreshButton.setText("")
        self.onTopButton.setText("")
    # retranslateUi

