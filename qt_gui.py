import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QFormLayout, QWidget, \
    QTabWidget, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox, QTextEdit, QRadioButton
from PyQt5 import QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Plotting
import matplotlib.pyplot as plt


class Window(QTabWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Windows
        self.create_tab = QWidget()
        self.solve_tab = QWidget()
        self.plot_tab = QWidget()
        self.summary_tab = QWidget()

        self.addTab(self.create_tab, "Create")
        self.addTab(self.solve_tab, "Solve")
        self.addTab(self.plot_tab, "Generations plots")
        self.addTab(self.summary_tab, "Summary")

        # Create Window
        self.x_start = QDoubleSpinBox()
        self.x_start.setRange(-1000000, 1000000)
        self.x_start.setValue(-2)
        self.x_end = QDoubleSpinBox()
        self.x_end.setRange(-1000000, 1000000)
        self.x_end.setValue(2)
        self.y_start = QDoubleSpinBox()
        self.y_start.setRange(-1000000, 1000000)
        self.y_start.setValue(-2)
        self.y_end = QDoubleSpinBox()
        self.y_end.setRange(-1000000, 1000000)
        self.y_end.setValue(2)
        self.function = QLineEdit()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Solve Window
        self.crossover_prob = QDoubleSpinBox()
        self.crossover_prob.setMinimum(0)
        self.crossover_prob.setMaximum(1)
        self.crossover_prob.setSingleStep(0.1)
        self.crossover_prob.setValue(0.6)
        self.mutation_prob = QDoubleSpinBox()
        self.mutation_prob.setMinimum(0)
        self.mutation_prob.setMaximum(1)
        self.mutation_prob.setSingleStep(0.1)
        self.mutation_prob.setValue(0.1)
        self.num_ind = QSpinBox()
        self.num_ind.setRange(2, 100)
        self.num_ind.setSingleStep(2)
        self.num_ind.setValue(10)
        self.num_gen = QSpinBox()
        self.num_gen.setRange(1, 100)
        self.num_gen.setValue(5)
        self.command_line = QTextEdit()
        self.command_line.setMinimumHeight(475)
        self.command_line.isReadOnly()
        self.generation_figure = plt.figure()
        self.generation_canvas = FigureCanvas(self.generation_figure)
        self.population_list = []
        self.current_gen = None
        self.min = QRadioButton("Min")
        self.min.toggled.connect(lambda: self.select_option(self.min))
        self.max = QRadioButton("Max")
        self.max.setChecked(True)
        self.max.toggled.connect(lambda: self.select_option(self.max))
        self.find_max = 1

        # Summary Window
        self.best_x = ""
        self.best_y = ""
        self.best_solution = float("nan")
        self.summary_figure = plt.figure()
        self.summary_canvas = FigureCanvas(self.summary_figure)
        self.mean = []
        self.median = []
        self.best = []
        self.best_x_label = QLabel("Best solution x: " + str(self.best_x))
        self.best_y_label = QLabel("y: " + str(self.best_y))
        self.best_f_label = QLabel("f(x,y): " + str(self.best_solution))

        # Initialize app windows
        self.create_tab_ui()
        self.solve_tab_ui()
        self.plot_tab_ui()
        self.summary_tab_ui()
        self.setWindowTitle("Genetic Algorithms")

    def create_tab_ui(self):

        button = QPushButton('Show function')
        toolbar = NavigationToolbar(self.canvas, self)
        button.clicked.connect(self.plot)
        data_form = QFormLayout()
        function_data = QHBoxLayout()
        function_data.addWidget(self.function)
        function_data.addWidget(QLabel("X=["))
        function_data.addWidget(self.x_start)
        function_data.addWidget(QLabel(","))
        function_data.addWidget(self.x_end)
        function_data.addWidget(QLabel("]"))
        function_data.addWidget(QLabel("  Y=["))
        function_data.addWidget(self.y_start)
        function_data.addWidget(QLabel(","))
        function_data.addWidget(self.y_end)
        function_data.addWidget(QLabel("]"))
        data_form.addRow("f(x,y) = ", function_data)
        data_form.addRow(toolbar)
        function_plot = QHBoxLayout()
        function_plot.addWidget(QLabel(""))
        function_plot.addWidget(self.canvas)
        function_plot.addWidget(QLabel(""))
        data_form.addRow(function_plot)
        data_form.addRow(button)

        self.create_tab.setLayout(data_form)

    def select_option(self, b):

        if b.text() == "Max":
            if b.isChecked() == True:
                self.find_max = 1
            else:
                self.find_max = 0

        if b.text() == "Min":
            if b.isChecked() == True:
                self.find_max = 0
            else:
                self.find_max = 1


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # main = Window()
    # main.show()

    secondary = QtWidgets.QMainWindow()
    uic.loadUi('GUI.ui', secondary)
    secondary.show()

    sys.exit(app.exec_())