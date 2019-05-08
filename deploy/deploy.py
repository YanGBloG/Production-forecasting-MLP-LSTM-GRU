import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from keras.models import load_model

from sklearn.metrics import r2_score, mean_squared_error

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PopupWindowScreen import ShowTrainLog

norm_font = QFont()
norm_font.setFamily("Helvetica")
norm_font.setPointSize(10)

large_font = QFont()
large_font.setFamily("Helvetica")
large_font.setPointSize(12)

class ModelDeploy(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(QMainWindow, self).__init__(*args, **kwargs)
        QMainWindow.setObjectName(self, "MainWindow")

        self.setMinimumSize(QSize(1345, 670))    
        self.setWindowTitle("Deploy Model")
        self.setWindowIcon(QIcon('touch.png'))
        self.setFont(norm_font)

        self.file_link = []
        self.model = []
        self.predicted_data = []

        self.statusBar()
        mainMenu = self.menuBar()
        
        Menu = mainMenu.addMenu('Menu')
        Open = Menu.addAction('Open')
        Save = Menu.addAction('Save')
        Exit = Menu.addAction('Exit')
        
        File = mainMenu.addMenu('File')
        Edit = mainMenu.addMenu('Edit')
        View = mainMenu.addMenu('View')
        Properties = mainMenu.addMenu('Properties')
        Help = mainMenu.addMenu('Help')

        self.evaluation_label = QLabel('Evaluation', self)
        self.evaluation_label.setGeometry(QRect(10, 23, 511, 21))
        self.evaluation_label.setStyleSheet('background-color: #DEF0D8;' 
        									'border-radius: 10px; color: #387144;')
        self.evaluation_label.setFont(large_font)
        self.evaluation_label.setAlignment(Qt.AlignCenter)

        self.evaluation_view = QGraphicsView(self)
        self.evaluation_view.setGeometry(QRect(10, 50, 511, 511))

        self.evaluation_figure = Figure()
        self.evaluation_figure_canvas = FigureCanvas(self.evaluation_figure)

        self.evaluation_figure_layout = QWidget(self)
        self.evaluation_figure_layout.setGeometry(QRect(11, 51, 509, 509))

        evaluation_box = QVBoxLayout(self.evaluation_figure_layout)
        evaluation_box.addWidget(self.evaluation_figure_canvas)

        self.result_label = QLabel('Plots', self)
        self.result_label.setGeometry(QRect(530, 23, 811, 21))
        self.result_label.setStyleSheet('background-color: rgb(0, 85, 255);'
        								'border-radius: 10px; color: white;')
        self.result_label.setFont(large_font)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.result_view = QGraphicsView(self)
        self.result_view.setGeometry(QRect(530, 50, 811, 591))

        self.result_figure = Figure()
        self.result_figure_canvas = FigureCanvas(self.result_figure)

        self.result_figure_layout = QWidget(self)
        self.result_figure_layout.setGeometry(QRect(531, 51, 809, 589))

        result_figure_box = QVBoxLayout(self.result_figure_layout)
        result_figure_box.addWidget(self.result_figure_canvas)

        self.line = QFrame(self)
        self.line.setGeometry(QRect(10, 560, 511, 16))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.load_trained_model = QPushButton('Load model', self)
        self.load_trained_model.setGeometry(QRect(201, 575, 101, 31))
        self.load_trained_model.clicked.connect(self.load_model_to_predict)

        self.load_new_data = QPushButton('Load data', self)
        self.load_new_data.setGeometry(QRect(311, 575, 101, 31))
        self.load_new_data.clicked.connect(self.load_predict_data)

        self.predict_data_button = QPushButton('Predict', self)
        self.predict_data_button.setGeometry(QRect(421, 575, 101, 31))
        self.predict_data_button.clicked.connect(self.model_predict)

        self.show_result_button = QPushButton('Show', self)
        self.show_result_button.setGeometry(QRect(201, 611, 101, 31))
        self.show_result_button.clicked.connect(self.plotEvaluation)
        self.show_result_button.clicked.connect(self.plotPredictResult)

        self.detail_button = QPushButton('Detail', self)
        self.detail_button.setGeometry(QRect(311, 611, 101, 31))
        self.detail_button.clicked.connect(self.showDetailResult)

        self.save_result_button = QPushButton('Save', self)
        self.save_result_button.setGeometry(QRect(421, 611, 101, 31))
        self.save_result_button.clicked.connect(self.SavePredictResult)

    def load_model_to_predict(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "",
        										"Keras model (*.h5)",
        										options=options)
        if fileName:
            model = load_model(fileName)
            if self.model:
                self.model = []
                self.model.append(model)
            else:
                self.model.append(model)

    def load_predict_data(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "",
        										"CSV Files (*.csv);;All Files (*.*)",
        										options=options)
        if fileName:
            if self.file_link:
                self.file_link = []
                self.file_link.append(fileName)
            else:
                self.file_link.append(fileName)

    def getPredictData(self):
        while True:
            try:
                data = pd.read_csv(self.file_link[0])
                X            = data.values[:, 1:-1]
                y            = data.values[:, -1]
                data['date'] = data['date'].astype('datetime64')
                date         = data['date']
                return X, y, date
            except:
                break

    def model_predict(self):
    	while True:
            try:
            	X_test, y_test, date = self.getPredictData()
            	y_pred = self.model[0].predict(X_test)
            	if not self.predicted_data:
            		self.predicted_data.append(y_pred)
            	else:
            		self.predicted_data = []
            		self.predicted_data.append(y_pred)
            	self.popupmsg('Predict completed', type_of_msg='info')
            	break
            except:
            	self.popupmsg('Something wrong. Load your test data or model first',
            		type_of_msg = 'warning')
            	break

    def plotEvaluation(self):
    	while True:
            try:
                X_test, y_test, date = self.getPredictData()
                y_pred = self.predicted_data[0].flatten()
                y_test = y_test.astype('float32')
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
                y_line = slope * y_test + intercept
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                fig = self.evaluation_figure.add_subplot(111)
                fig.clear()
                fig.scatter(y_test, y_pred, color='#2C89BE', zorder=2, 
                			label='$R^2$ = {:.6f}'.format(r2))
                fig.plot(y_test, y_line, zorder=1, label='$RMSE$ = {:.6f}'.format(rmse))
                fig.set_xlabel('Original')
                fig.set_ylabel('Prediction')
                fig.legend(markerscale=0, handlelength=0, loc='best')
                self.evaluation_figure_canvas.draw()
                break
            except:
            	self.popupmsg("Something wrong with your process!!!",
            				type_of_msg='warning')

    def plotPredictResult(self):
    	while True:
            try:
                X_test, y_test, date = self.getPredictData()
                y_pred = self.predicted_data[0]
                t = pd.DataFrame({'date': date,
                                'x': y_test,
                                'y': y_pred.flatten()})

                t.sort_values(by=['date'], inplace=True)

                fig = self.result_figure.add_subplot(111)
                fig.clear()
                fig.plot(t['date'], t['x'], label='Original data')
                fig.plot(t['date'], t['y'], label='Predict data')
                fig.set_xlabel("Date")
                fig.set_ylabel("Flow rate")
                fig.legend(loc="best")
                self.result_figure_canvas.draw()
                break
            except:
                self.popupmsg("Make predict on your data first",
                    type_of_msg = "warning")
                break

    def showDetailResult(self):
    	while True:
            try:
                X_test, y_test, date = self.getPredictData()
                y_pred = self.predicted_data[0].flatten()

                df = pd.DataFrame({'Date': date,
    							   'Original': y_test,
    							   'Predict': y_pred})

                df.sort_values(by=['Date'], inplace=True)

                self.detail_result = ShowTrainLog(df)        
                self.detail_result.show()
                break
            except:
                self.popupmsg("Make prediction on your data first",
                			  type_of_msg="warning")
                break

    def SavePredictResult(self):
        X_test, y_test, date = self.getPredictData()
        y_pred = self.predicted_data[0].flatten()

        df = pd.DataFrame({'Date': date,
        				   'Original': y_test,
        				   'Predict': y_pred})

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save file', '',
        										  'CSV file (*.csv)', 
        										  options=options)
        if fileName:
            df.to_csv(fileName, index=False)

    @pyqtSlot()
    def popupmsg(self, msg, type_of_msg = "warning"):
        if type_of_msg == 'info':
            QMessageBox.information(self, 'Information!!!', msg, 
            						QMessageBox.Ok,
            						QMessageBox.Ok)
        elif type_of_msg == 'warning':
            QMessageBox.warning(self, 'Warning!!!', msg, 
            					QMessageBox.Ok,
            					QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ModelDeploy()
    mainWin.show()
    sys.exit(app.exec_())