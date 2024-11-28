from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,
    QComboBox, QTableWidget, QTableWidgetItem, QTabWidget, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from src.data.dataset_handler import DatasetHandler
from src.models.trainer import ModelTrainer

class AIModelApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Model Generator Framework")
        self.setGeometry(100, 100, 1000, 800)

        self.dataset_handler = None
        self.model_trainer = None

        # Layout and tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.init_dataset_tab()
        self.init_visualization_tab()
        self.init_model_tab()

    def init_dataset_tab(self):
        """Create the dataset tab."""
        self.dataset_tab = QWidget()
        self.tabs.addTab(self.dataset_tab, "Dataset")

        layout = QVBoxLayout()
        self.dataset_tab.setLayout(layout)

        # Label for displaying dataset file info
        self.label = QLabel("Select a dataset to begin.")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Load Dataset Button
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_button)

        # Preprocessing Options
        layout.addWidget(QLabel("Preprocessing:"))

        # Normalize Button
        self.normalize_button = QPushButton("Normalize Data")
        self.normalize_button.clicked.connect(self.normalize_data)
        layout.addWidget(self.normalize_button)

        # Sliding Window Parameters
        self.sliding_window_label = QLabel("Sliding Window Parameters:")
        layout.addWidget(self.sliding_window_label)

        self.window_size_input = QLineEdit()
        self.window_size_input.setPlaceholderText("Window Size (e.g., 50)")
        layout.addWidget(self.window_size_input)

        self.step_size_input = QLineEdit()
        self.step_size_input.setPlaceholderText("Step Size (e.g., 10)")
        layout.addWidget(self.step_size_input)

        self.apply_window_button = QPushButton("Apply Sliding Window")
        self.apply_window_button.clicked.connect(self.apply_sliding_window)
        layout.addWidget(self.apply_window_button)

        # Target Column Selector
        self.target_selector = QComboBox()
        layout.addWidget(self.target_selector)

        # Table view for displaying dataset
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)

    def normalize_data(self):
        """Normalize the dataset."""
        if not self.dataset_handler or self.dataset_handler.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        self.dataset_handler.normalize_data()
        QMessageBox.information(self, "Success", "Data normalized successfully.")

    def apply_sliding_window(self):
        """Apply sliding window to the dataset."""
        if not self.dataset_handler or self.dataset_handler.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        try:
            window_size = int(self.window_size_input.text())
            step_size = int(self.step_size_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid integers for window and step sizes.")
            return

        windows = self.dataset_handler.apply_sliding_window(window_size, step_size)
        QMessageBox.information(self, "Success", f"Sliding window applied. Generated {len(windows)} samples.")

    def init_visualization_tab(self):
        """Create the visualization tab."""
        self.visualization_tab = QWidget()
        self.tabs.addTab(self.visualization_tab, "Visualization")

        layout = QVBoxLayout()
        self.visualization_tab.setLayout(layout)

        # Dropdown for selecting type of plot
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(["Scatter Plot", "3D Plot", "Time-Series Plot"])
        layout.addWidget(QLabel("Select Plot Type:"))
        layout.addWidget(self.plot_selector)

        # Plot Button
        self.plot_button = QPushButton("Generate Plot")
        self.plot_button.clicked.connect(self.generate_plot)
        layout.addWidget(self.plot_button)

        # Scatter plot button
        self.scatter_button = QPushButton("Scatter Plot")
        self.scatter_button.clicked.connect(self.scatter_plot)
        layout.addWidget(self.scatter_button)

        # Histogram button
        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.clicked.connect(self.histogram_plot)
        layout.addWidget(self.histogram_button)

        # Correlation heatmap button
        self.heatmap_button = QPushButton("Correlation Heatmap")
        self.heatmap_button.clicked.connect(self.heatmap_plot)
        layout.addWidget(self.heatmap_button)

        # Canvas for Matplotlib plots
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def init_model_tab(self):
        self.model_tab = QWidget()
        self.tabs.addTab(self.model_tab, "Model Training")
        layout = QVBoxLayout()
        self.model_tab.setLayout(layout)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Linear Regression", "Decision Tree", "Random Forest", "Neural Network", "RNN", "LSTM"])
        layout.addWidget(QLabel("Select Model Type:"))
        layout.addWidget(self.model_selector)

        self.epochs_input = QLineEdit()
        self.epochs_input.setPlaceholderText("Enter epochs (for Neural Network)")
        layout.addWidget(self.epochs_input)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.label.setText(f"Loaded Dataset: {file_path}")
            self.dataset_handler = DatasetHandler(file_path)
            self.dataset_handler.load_data()

            # Populate the target column dropdown
            self.target_selector.clear()
            self.target_selector.addItems(self.dataset_handler.data.columns)

            # Display dataset in the table view
            self.display_dataset()

    def display_dataset(self):
        """Display the dataset in the QTableWidget."""
        if self.dataset_handler and self.dataset_handler.data is not None:
            data = self.dataset_handler.data
            self.data_table.setRowCount(len(data))
            self.data_table.setColumnCount(len(data.columns))
            self.data_table.setHorizontalHeaderLabels(data.columns)

            for i, row in data.iterrows():
                for j, value in enumerate(row):
                    self.data_table.setItem(i, j, QTableWidgetItem(str(value)))

    def generate_plot(self):
        """Generate the selected plot."""
        if not self.dataset_handler or self.dataset_handler.data is None:
            self.label.setText("Please load a dataset first.")
            return

        plot_type = self.plot_selector.currentText()
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d' if plot_type == "3D Plot" else None)

        if plot_type == "Scatter Plot":
            target_column = self.target_selector.currentText()
            feature = self.dataset_handler.X.columns[0]
            ax.scatter(self.dataset_handler.data[feature], self.dataset_handler.data[target_column])
        elif plot_type == "3D Plot":
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            if "Gyroscope_X" in self.dataset_handler.data.columns:
                ax.scatter(
                    self.dataset_handler.data["Gyroscope_X"],
                    self.dataset_handler.data["Gyroscope_Y"],
                    self.dataset_handler.data["Gyroscope_Z"],
                    c='r'
                )
        elif plot_type == "Time-Series Plot":
            target_column = self.target_selector.currentText()
            ax.plot(self.dataset_handler.data[target_column])

        self.canvas.draw()

    def scatter_plot(self):
        """Generate a scatter plot for the selected target column."""
        if not self.dataset_handler:
            self.label.setText("Please load a dataset first.")
            return

        target_column = self.target_selector.currentText()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if len(self.dataset_handler.X.columns) > 1:
            feature = self.dataset_handler.X.columns[0]
            ax.scatter(self.dataset_handler.data[feature], self.dataset_handler.data[target_column])
            ax.set_title(f"Scatter Plot: {feature} vs {target_column}")
            ax.set_xlabel(feature)
            ax.set_ylabel(target_column)
        else:
            self.label.setText("Insufficient data for scatter plot.")

        self.canvas.draw()

    def histogram_plot(self):
        """Generate a histogram for the selected target column."""
        if not self.dataset_handler:
            self.label.setText("Please load a dataset first.")
            return

        target_column = self.target_selector.currentText()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.hist(self.dataset_handler.data[target_column], bins=20, alpha=0.7, color="blue")
        ax.set_title(f"Histogram: {target_column}")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Frequency")

        self.canvas.draw()

    def heatmap_plot(self):
        """Generate a heatmap for the dataset correlations."""
        if not self.dataset_handler:
            self.label.setText("Please load a dataset first.")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        correlation_matrix = self.dataset_handler.data.corr()
        cax = ax.matshow(correlation_matrix, cmap="coolwarm")
        self.figure.colorbar(cax)

        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticklabels(correlation_matrix.columns)

        self.canvas.draw()

    def train_model(self):
        """Train the selected model with preprocessed data."""
        if not self.dataset_handler:
            self.label.setText("Please load and preprocess the dataset first.")
            return

        target_column = self.target_selector.currentText()
        self.dataset_handler.preprocess(target_column)
        X_train, X_test, y_train, y_test = self.dataset_handler.get_splits()

        model_type = self.model_selector.currentText()
        self.model_trainer = ModelTrainer()

        if model_type == "Neural Network":
            epochs = int(self.epochs_input.text()) if self.epochs_input.text() else 10
            self.model_trainer.train_neural_network(X_train, y_train, epochs)
        elif model_type == "Random Forest":
            self.model_trainer.train_random_forest(X_train, y_train)
        elif model_type == "Decision Tree":
            self.model_trainer.train_decision_tree(X_train, y_train)
        elif model_type == "Linear Regression":
            self.model_trainer.train_linear_regression(X_train, y_train)
        elif model_type == "RNN":
            self.model_trainer.train_rnn(X_train, y_train, epochs)
        elif model_type == "LSTM":
            self.model_trainer.train_lstm(X_train, y_train, epochs)

        results = self.model_trainer.evaluate(X_test, y_test)
        self.label.setText(f"Training complete. Results: {results}")
