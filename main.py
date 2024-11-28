import sys
import os
from PyQt5.QtWidgets import QApplication
from src.gui.app import AIModelApp  # Importing the main GUI class

# Add the `src` folder to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def main():
    """Entry point for the AI Model Generator Framework."""
    app = QApplication(sys.argv)  # Initialize the PyQt application
    main_window = AIModelApp()    # Create an instance of the GUI
    main_window.show()            # Show the main window
    sys.exit(app.exec_())         # Start the application event loop

if __name__ == "__main__":
    main()
