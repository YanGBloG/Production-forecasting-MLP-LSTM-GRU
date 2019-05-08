import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from math import sin, pi

import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

# from nxZoomArchitecture import DrawNN
# from nxZoomArchitecture import NeuralNetwork as nn

from PandasModel import PandasModel

# import ImagesResources

norm_font = QFont()
norm_font.setFamily("Helvetica")
norm_font.setPointSize(10)

large_font = QFont()
large_font.setFamily("Helvetica")
large_font.setPointSize(12)
        

# class PopupWindow(QMainWindow):
#     def __init__(self, neurons_and_layers_list, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._main = QWidget()
#         self.setCentralWidget(self._main)
#         self.layoutScaled = QVBoxLayout(self._main)
        
#         self.setMinimumSize(QSize(1200, 650))    
#         self.setWindowTitle("Fully connected layers")
#         self.setWindowIcon(QIcon(':/fig/neuralnetwork.png'))

#         self.neurons_and_layers_list = neurons_and_layers_list

# ### FIGURE
#         self.network_architect_canvas = FigureCanvas(Figure())
#         self.layoutScaled.addWidget(self.network_architect_canvas)

#         self.addToolBar(Qt.BottomToolBarArea, NavigationToolbar(self.network_architect_canvas, self))
            
#         m = self.makeNetworkArchitecture()

#     def makeConfig(self):
#         conditions = len(self.neurons_and_layers_list)
#         if conditions == 10:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3], self.neurons_and_layers_list[4], self.neurons_and_layers_list[5], self.neurons_and_layers_list[6], self.neurons_and_layers_list[7], self.neurons_and_layers_list[8], self.neurons_and_layers_list[9]]
#         elif conditions == 9:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3], self.neurons_and_layers_list[4], self.neurons_and_layers_list[5], self.neurons_and_layers_list[6], self.neurons_and_layers_list[7], self.neurons_and_layers_list[8]]
#         elif conditions == 8:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3], self.neurons_and_layers_list[4], self.neurons_and_layers_list[5], self.neurons_and_layers_list[6], self.neurons_and_layers_list[7]]
#         elif conditions == 7:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3], self.neurons_and_layers_list[4], self.neurons_and_layers_list[5], self.neurons_and_layers_list[6]]            
#         elif conditions == 6:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3], self.neurons_and_layers_list[4], self.neurons_and_layers_list[5]]
#         elif conditions == 5:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3], self.neurons_and_layers_list[4]]
#         elif conditions == 4:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2], self.neurons_and_layers_list[3]]
#         elif conditions == 3:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1], self.neurons_and_layers_list[2]]
#         elif conditions == 2:
#             return [self.neurons_and_layers_list[0], self.neurons_and_layers_list[1]]
        
#     def makeNetworkArchitecture(self):
#         neurals_and_layers = self.makeConfig()
#         network_config = DrawNN(neurals_and_layers)
#         widest_layer = network_config.getWidestLayer()
#         network = nn(widest_layer)
#         full_config = network_config.nn_config()
        
#         for l in full_config:
#             network.add_layer(l)

#         ax = self.network_architect_canvas.figure.subplots()
#         plt.clf()
#         network.nn_draw(ax)
#         ax.axis('equal')
# ##        ax.axis('scaled')
#         ax.axis('off')
#         self.network_architect_canvas.draw()

class ShowTrainLog(QWidget):
    def __init__(self, dataframe, parent=None):
        QWidget.__init__(self, parent=None)

        self.setMinimumSize(QSize(370, 600))    
        self.setWindowTitle("Detail view")
        self.setWindowIcon(QIcon('calculate.png'))

        self.dataframe = dataframe

        vLayout = QVBoxLayout(self)
        hLayout = QHBoxLayout()
        vLayout.addLayout(hLayout)
        self.pandasTv = QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.pandasTv.setSortingEnabled(True)
        df = self.dataframe
        model = PandasModel(df)
        self.pandasTv.setModel(model)

class GetInputValue(QWidget):
    def __init__(self, dataframe, parent=None):
        QWidget.__init__(self, parent=None)

        self.setMinimumSize(QSize(370, 600))    
        self.setWindowTitle("Detail view")
        self.setWindowIcon(QIcon('calculate.png'))

        self.dataframe = dataframe

        vLayout = QVBoxLayout(self)
        hLayout = QHBoxLayout()
        vLayout.addLayout(hLayout)
        self.pandasTv = QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.pandasTv.setSortingEnabled(True)
        df = self.dataframe
        model = PandasModel(df)
        self.pandasTv.setModel(model)
        
        
##if __name__ == "__main__":
##    app = QApplication(sys.argv)
##    mainWin = PopupWindow()
##    mainWin.show()
##    sys.exit(app.exec_())
