from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QMessageBox, QDialog, QDialogButtonBox, QSlider, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np
import sys
import PIL.Image
from pathlib import Path
from image_processing import Histogram, Brightness, Conversion, Binarization, GrayscaleConversionError, Filter

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.source_image = SourceImage()

    def init_ui(self):
        #set window title
        self.setWindowTitle("Podstawy biometrii")

        #prepare image canvas
        self.image_layout = QtWidgets.QVBoxLayout(self.matplotlibEmbedWidget)
        self.image_canvas = FigureCanvas(Figure())
        self.image_layout.addWidget(NavigationToolbar(self.image_canvas, self))
        self.image_layout.addWidget(self.image_canvas)
        self._image_ax = self.image_canvas.figure.subplots()

        #prepare histogram canvas
        self.histogram_layout = QtWidgets.QVBoxLayout(self.histogramWidget)
        self.histogram_canvas = FigureCanvas(Figure())
        self.histogram_layout.addWidget(NavigationToolbar(self.histogram_canvas, self))
        self.histogram_layout.addWidget(self.histogram_canvas)
        self._histogram_ax = self.histogram_canvas.figure.subplots()

        #prepare image canvas mouse events
        self.mouse_move_connection_id = self.image_canvas.mpl_connect('motion_notify_event', self.on_move)
        self.image_canvas.mpl_connect('button_press_event', self.on_click)
        self.image_canvas.mpl_connect('figure_leave_event', self.on_leave_figure)
        
        #set r,g,b validation check events
        self.r_value.editingFinished.connect(lambda: self.on_color_change(self.r_value))
        self.g_value.editingFinished.connect(lambda: self.on_color_change(self.g_value))
        self.b_value.editingFinished.connect(lambda: self.on_color_change(self.b_value))
        
        #set menu actions
        self.actionOpen.triggered.connect(self.load_image)
        self.actionSave.triggered.connect(self.save_image)
        self.actionExit.triggered.connect(self.close_app)
        self.actionNormalize.triggered.connect(self.normalize_histogram)
        self.actionEqualizeGrayscale.triggered.connect(self.equalize_histogram_grayscale)
        self.actionEqualizeYCrCb.triggered.connect(self.equalize_histogram_YCrCb)
        self.actionBrightness.triggered.connect(self.brighten)
        self.actionGrayscale.triggered.connect(self.grayscale_conversion)
        self.actionOtsu.triggered.connect(self.otsu_binarization)
        self.actionBinary_Thresholding.triggered.connect(self.binary_thresholding)
        self.actionNiblack.triggered.connect(self.niblack_binarization)
        self.actionLinear_Filter.triggered.connect(self.linear_filter)
        self.actionKuwahara_Filter.triggered.connect(self.kuwahara_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionBox_Blur.triggered.connect(self.box_blur_filter)
        self.actionGaussian_Blur.triggered.connect(self.gaussian_blur_filter)


        #set button action
        self.update_btn.clicked.connect(self.change_rgb_value)

        #set histogram checkboxes actions
        self.r_hist_checkbox.stateChanged.connect(lambda: self.on_checkbox_state_change(self.r_hist_checkbox))
        self.g_hist_checkbox.stateChanged.connect(lambda: self.on_checkbox_state_change(self.g_hist_checkbox))
        self.b_hist_checkbox.stateChanged.connect(lambda: self.on_checkbox_state_change(self.b_hist_checkbox))
        self.avg_hist_checkbox.stateChanged.connect(lambda: self.on_checkbox_state_change(self.avg_hist_checkbox))
        
    def load_image(self):
        try:
            name, filter = QtWidgets.QFileDialog.getOpenFileName(None, "Open image file", "", "Image files (*.png *.jpg *.gif *tif *tiff *jpeg *bmp)")
            #load image
            if name is not None:
                self.source_image.path_to_image = Path(name)
                self.source_image.image_suffix = self.source_image.path_to_image.suffix
                self.source_image.image_stem = self.source_image.path_to_image.stem
                self.source_image.img=PIL.Image.open(self.source_image.path_to_image)
                self.source_image.img = self.source_image.img.convert ('RGB')
                self.source_image.image_pixelmap = self.source_image.img.load()
                self._image_ax.clear()
                self._image_ax.imshow(self.source_image.img)
                self.image_canvas.draw()
        except FileNotFoundError as fnfe:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("File not found")
            msg.setInformativeText(str(fnfe))
            msg.setWindowTitle("File error")
            msg.exec_()
        except PermissionError as pe:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Permision error")
            msg.setInformativeText(str(pe))
            msg.setWindowTitle("File error")
            msg.exec_()

    def save_image(self):
        if(self.source_image.image_suffix is not None and self.source_image.image_stem is not None):
            try:
                image_name = self.source_image.image_stem+'_edit'
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save as... File', image_name , filter="png file (*.png);;jpg file (*.jpg);;jpeg file (*.jpeg);;gif file (*.gif);;tif file (*.tif);;tiff file (*.tiff);;bmp file (*.bmp)")
                if filename is not None:
                    try:
                        self.source_image.img.save(Path(filename))
                    except AttributeError:
                        PIL.Image.fromarray(self.source_image.img).save(Path(filename))
            except ValueError:
                pass
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No file to save")
            msg.setInformativeText("Please load your file first")
            msg.setWindowTitle("Saving issue")
            msg.exec_()

    def update_image(self):
        self._image_ax.clear()
        self._image_ax.imshow(self.source_image.img, cmap = self.source_image.cmap)
        self.image_canvas.draw()

    def draw_histogram(self):
        r_channel_hist = Histogram.compute_histogram(self.source_image.img, [0])
        g_channel_hist = Histogram.compute_histogram(self.source_image.img, [1])
        b_channel_hist = Histogram.compute_histogram(self.source_image.img, [2])
        avg_hist = (r_channel_hist + g_channel_hist + b_channel_hist)/3
        self._histogram_ax.clear()
        self._histogram_ax.set_xlim([0, 256])
        x = [i for i in range(r_channel_hist.ravel().shape[0])]
        y = [0 for i in range(r_channel_hist.ravel().shape[0])]

        if self.r_hist_checkbox.isChecked():
            self._histogram_ax.plot(r_channel_hist, 'r-.', alpha = 0.4)
            self._histogram_ax.fill_between(x, r_channel_hist.ravel(), y, facecolor='red', alpha=0.2)
        
        if self.g_hist_checkbox.isChecked():
            self._histogram_ax.plot(g_channel_hist, 'g-.', alpha = 0.4)
            self._histogram_ax.fill_between(x, g_channel_hist.ravel(), y, facecolor='green', alpha=0.2)
        
        if self.b_hist_checkbox.isChecked():
            self._histogram_ax.plot(b_channel_hist, 'b-.', alpha = 0.4)
            self._histogram_ax.fill_between(x, b_channel_hist.ravel(), y, facecolor='blue', alpha=0.2)
        
        if self.avg_hist_checkbox.isChecked():
            self._histogram_ax.plot(avg_hist, 'k-', alpha = 0.4)
            self._histogram_ax.fill_between(x, avg_hist.ravel(), y, facecolor='black', alpha=0.2)
        self.histogram_canvas.draw()

    def normalize_histogram(self):
        dlg = NormalizeDialog(self)
        if dlg.exec_():
            a, b = dlg.get_values()
            self.source_image.img = Histogram.normalize_histogram(self.source_image.img, a, b)

    def equalize_histogram_grayscale(self):
        self.source_image.img = Histogram.equalize_histogram_grayscale(self.source_image.img)

    def equalize_histogram_YCrCb(self):
        self.source_image.img = Histogram.equalize_histogram_YCrCb(self.source_image.img)
    
    def brighten(self):
        dlg = BrightnessDialog(self)
        if dlg.exec_():
            gamma = dlg.get_values()
            self.source_image.img = Brightness.gamma_correction(self.source_image.img, gamma)

    def grayscale_conversion(self):
        self.source_image.img = Conversion.convert_2_gray(self.source_image.img)
        self.source_image.cmap = "gray"
        self.update_image()
        
    def otsu_binarization(self):
        try:
            self.source_image.img = Binarization.otsu(self.source_image.img)
        except GrayscaleConversionError as te:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText(str(te))
                msg.setWindowTitle("Invalid Image conversion")
                msg.exec_()

    def niblack_binarization(self):
        dlg = NiblackDialog(self)
        if dlg.exec_():
            kernel, k = dlg.get_values()
            try:
                self.source_image.img = Binarization.niblack(self.source_image.img, kernel, k)
            except GrayscaleConversionError as te:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText(str(te))
                msg.setWindowTitle("Invalid Image conversion")
                msg.exec_()

    def binary_thresholding(self):
        dlg = BinaryThresholdingDialog(self)
        if dlg.exec_():
            thresh = dlg.get_values()
            try:
                self.source_image.img = Binarization.binary_thresholding(self.source_image.img, thresh)
            except GrayscaleConversionError as te:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText(str(te))
                msg.setWindowTitle("Invalid Image conversion")
                msg.exec_()

    def linear_filter(self):
        dlg = LinearFilterDialog(self)
        if dlg.exec_():
            kernel = dlg.get_values()
            self.source_image.img = Filter.linear_filter(self.source_image.img, kernel)

    def kuwahara_filter(self):
        dlg = KuwaharaFilterDialog(self)
        if dlg.exec_():
            kernel = dlg.get_values()
            self.source_image.img = Filter.kuwahara_filter(self.source_image.img, kernel)


    def median_filter(self):
        dlg = MedianFilterDialog(self)
        if dlg.exec_():
            kernel = dlg.get_values()
            self.source_image.img = Filter.median(self.source_image.img, kernel)

    def box_blur_filter(self):
        self.source_image.img = Filter.box_blur(self.source_image.img)

    def gaussian_blur_filter(self):
        self.source_image.img = Filter.gaussian_blur(self.source_image.img)

    def close_app(self):
        sys.exit(app.exec_())

    def on_move(self, event):
        if event.inaxes:
            ax = event.inaxes  # the axes instance
            self.source_image.x = int(round(event.xdata))
            self.source_image.y = int(round(event.ydata))
            if self.source_image.image_pixelmap != None:

                r,g,b = tuple(self.source_image.image_pixelmap[self.source_image.x, self.source_image.y])
                self.r_value.setText(str(r))
                self.g_value.setText(str(g))
                self.b_value.setText(str(b))

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            self.image_canvas.mpl_disconnect(self.mouse_move_connection_id)

    def on_leave_figure(self, event):
        self.mouse_move_connection_id = self.image_canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_color_change(self, obj=None):
        if obj != None:
            try:
                value = int(obj.text())
                if value>255:
                    obj.setText("255")
                elif value<0:
                    obj.setText("0")
                else:
                    obj.setText(str(value))
            except ValueError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Value need to be number")
                msg.setInformativeText('of range 0 to 255')
                msg.setWindowTitle("Invalid input")
                msg.exec_()

    def on_checkbox_state_change(self, obj=None):
        if obj != None:
            self.draw_histogram()
            
    def change_rgb_value(self):
        if(self.x != None or self.y != None):
            try:
                r = int(self.r_value.text())
                g = int(self.g_value.text())
                b = int(self.b_value.text())
                self.source_image.image_pixelmap[self.source_image.x, self.source_image.y] = r, g, b
                self._image_ax.imshow(self.source_image.img, cmap = self.source_image.cmap )
                self.image_canvas.draw()

            except ValueError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Invalid RGB values")
                msg.setWindowTitle("Invalid RGB values")
                msg.exec_()


                
class SourceImage:
    """description of class"""
    def __init__(self, *args, **kwargs):

        #initialization of image
        self._img = None
        self.image_pixelmap = None
        
        #image file information
        self.image_stem = None
        self.image_suffix = None 
        self.path_to_image = None

        #image coordinates
        self.x = None
        self.y = None
    
        #image cmap
        self.cmap = None
    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self._img = value
        MainWindow.update_image(main)
        # update histogram each time the image is changed
        MainWindow.draw_histogram(main)

class NavigationToolbar(NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Forward', 'Back', 'Zoom')]

class NormalizeDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(NormalizeDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Normalize histogram")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()
        

        self.low_label = QLabel()
        self.low_label.setText("Lower value:")
        self.low_input  = QSpinBox()
        self.low_input.setRange(0, 255)
        self.low_input.setValue(0)

        self.high_label = QLabel()
        self.high_label.setText("Higher value:")
        self.high_input  = QSpinBox()
        self.high_input.setRange(0, 255)
        self.high_input.setValue(255)

        self.high_input.valueChanged.connect(self.validate)
        self.low_input.valueChanged.connect(self.validate)

        self.input_layout.addWidget(self.low_label)
        self.input_layout.addWidget(self.low_input)
        self.input_layout.addWidget(self.high_label)
        self.input_layout.addWidget(self.high_input)

        self.layout.addLayout(self.input_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def validate(self):
        if self.low_input.value()>=self.high_input.value():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Lower value cannot be equal or higher than higher value!")
            msg.setWindowTitle("Invalid values")
            msg.exec_()
            self.low_input.setValue(0)
            self.high_input.setValue(255)

    def get_values(self):
        return self.low_input.value(), self.high_input.value()

class BrightnessDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(BrightnessDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Brighten image")
        self.setMinimumWidth(600)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()
        

        self.label = QLabel()
        self.label.setText("Gamma value:")
        
        self.input = QSlider(Qt.Horizontal)
        self.input.setMinimum(0)
        self.input.setMaximum(2500)
        self.input.setValue(100)

        self.input_value = QLabel()
        self.input_value.setText("1.00")

        self.input_layout.addWidget(self.label)
        self.input_layout.addWidget(self.input)
        self.input_layout.addWidget(self.input_value)
        
        self.input.valueChanged.connect(self.update)

        self.layout.addLayout(self.input_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def update(self):
        self.input_value.setText(str(self.input.value()/100))

    def get_values(self):
        return self.input.value()/100

class BinaryThresholdingDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(BinaryThresholdingDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Binary thresholding")
        self.setMinimumWidth(600)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()
        

        self.label = QLabel()
        self.label.setText("Thresh:")
        
        self.input = QSlider(Qt.Horizontal)
        self.input.setMinimum(0)
        self.input.setMaximum(255)
        self.input.setValue(127)

        self.input_value = QLabel()
        self.input_value.setText("127")

        self.input_layout.addWidget(self.label)
        self.input_layout.addWidget(self.input)
        self.input_layout.addWidget(self.input_value)
        
        self.input.valueChanged.connect(self.update)

        self.layout.addLayout(self.input_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def update(self):
        self.input_value.setText(str(self.input.value()))

    def get_values(self):
        return self.input.value()

class NiblackDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(NiblackDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Niblack Threshold")
        self.setMinimumWidth(600)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()
        

        self.label = QLabel()
        self.label.setText("Kernel:")
        
        self.input = QSpinBox()
        self.input.setMinimum(3)
        self.input.setMaximum(49)
        self.input.setValue(25)
        self.input.setSingleStep(2)

        self.label_2 = QLabel()
        self.label_2.setText("K:")

        self.input_2 = QDoubleSpinBox()
        self.input_2.setMinimum(-1)
        self.input_2.setMaximum(1)
        self.input_2.setValue(0.6)
        self.input_2.setSingleStep(0.1)

        self.input_layout.addWidget(self.label)
        self.input_layout.addWidget(self.input)
        self.input_layout.addWidget(self.label_2)
        self.input_layout.addWidget(self.input_2)
        
        
        self.layout.addLayout(self.input_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def get_values(self):
        return self.input.value(), self.input_2.value()

class LinearFilterDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(LinearFilterDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Linear filter")
        self.setMinimumWidth(600)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()
        
        self.kernel_label = QLabel()
        self.kernel_label.setText("Kernel:")

        self.filter_label = QLabel()
        self.filter_label.setText("Filter:")
        
        self.kernel_layout = QtWidgets.QVBoxLayout()
        self.kernel_grid = QtWidgets.QGridLayout()

        self.inputs = {}

        for i in range(3):
            for j in range(3):
                # keep a reference to the buttons
                self.inputs[(i, j)] = QSpinBox()
                self.inputs[(i, j)].setMinimum(-100)
                self.inputs[(i, j)].setMaximum(100)
                # add to the layout
                self.kernel_grid.addWidget(self.inputs[(i, j)], i, j)

        self.option_layout = QtWidgets.QVBoxLayout()
        self.option = QtWidgets.QComboBox()
        
        self.option.addItems(['Default',
                              'Prewitt 0 deg',
                              'Prewitt 45 deg',
                              'Prewitt 90 deg',
                              'Prewitt 135 deg',
                              'Prewitt 180 deg',
                              'Prewitt 225 deg',
                              'Prewitt 270 deg',
                              'Prewitt 315 deg',
                              'Sobel 0 deg',
                              'Sobel 45 deg',
                              'Sobel 90 deg',
                              'Sobel 135 deg',
                              'Sobel 180 deg',
                              'Sobel 225 deg',
                              'Sobel 270 deg',
                              'Sobel 315 deg',
                              'Laplace 1',
                              'Laplace 2',
                              'Laplace 3',
                              'Edge Detection 1',
                              'Edge Detection 2',
                              'Edge Detection 3',
                              'Edge Detection 4',
                              ])

        self.kernel_layout.addWidget(self.kernel_label)
        self.kernel_layout.addLayout(self.kernel_grid)

        self.option_layout.addWidget(self.filter_label)
        self.option_layout.addWidget(self.option)

        self.input_layout.addLayout(self.kernel_layout)
        self.input_layout.addLayout(self.option_layout)
        
        self.option.currentIndexChanged.connect(self.update)

        self.layout.addLayout(self.input_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def update(self):
        for i in range(3):
            for j in range(3):
                # keep a reference to the buttons
                self.inputs[(i, j)].setValue(Filter.filters[self.option.currentIndex(),i,j]) 

    def get_values(self):
        output = [[self.inputs[(i, j)].value() for i in range(3)] for j in range(3)] 
        return np.array(output, np.float32)

class MedianFilterDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(MedianFilterDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Median filter")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()

        self.groupBox = QtWidgets.QGroupBox('Kernel size')
        self.radio_3 = QtWidgets.QRadioButton('3')
        self.radio_3.value = 3
        self.radio_3.setChecked(True)
        self.value = 3

        self.radio_5 = QtWidgets.QRadioButton('5')
        self.radio_5.value = 5

        self.radio_3.toggled.connect(lambda:self.update(self.radio_3))
        self.radio_5.toggled.connect(lambda:self.update(self.radio_5))
        
        self.input_layout.addWidget(self.radio_3)
        self.input_layout.addWidget(self.radio_5)

        self.groupBox.setLayout(self.input_layout)

        self.layout.addWidget(self.groupBox)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def update(self, obj):
        self.value = obj.value

    def get_values(self):
        return self.value                


class KuwaharaFilterDialog(QDialog):

    def __init__(self, img,  *args, **kwargs):
        super(KuwaharaFilterDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Kuwahara filter")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QHBoxLayout()

        self.groupBox = QtWidgets.QGroupBox('Kernel size')
        
        self.input = QSpinBox()
        self.input.setMinimum(3)
        self.input.setMaximum(49)
        self.input.setValue(5)
        self.input.setSingleStep(2)

        self.input_layout.addWidget(self.input)
        self.groupBox.setLayout(self.input_layout)

        self.layout.addWidget(self.groupBox)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def get_values(self):
        return self.input.value()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = MainWindow()
    uic.loadUi('main.ui', main)
    main.init_ui()
    main.show()
    sys.exit(app.exec_())