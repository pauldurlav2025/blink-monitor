import json
import logging
import os
import psutil
import signal
import sqlite3
import sys
import threading
import time

import cv2
import numpy as np

from collections import deque
from datetime import datetime
from face_analyzer import FaceAnalyzer, find_working_camera, configure_logging, signal_handler, setup_signal_handler
from face_analyzer.config import FRAME_WIDTH, FRAME_HEIGHT, FPS
from face_analyzer.monitoring import metrics_monitor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plotter

from PyQt5.QtWidgets import QSplashScreen, QColorDialog, QMenu, QToolButton, QWidgetAction, QGridLayout, QSizePolicy, QTableWidgetItem, QHeaderView, QTableWidget, QComboBox, QProgressBar, QMessageBox, QListWidgetItem, QListWidget, QSpinBox, QTabWidget, QRadioButton, QDialog, QDateTimeEdit, QGroupBox, QCheckBox, QTextBrowser, QLabel, QApplication, QPushButton, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QMainWindow, QAction, QScrollArea, QSplitter, QFileDialog, QFormLayout 
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QSettings, QPoint, QUrl
from PyQt5.QtGui import QIcon, QColor, QFont, QPalette, QPixmap

PROG_PATH               = __file__.replace(os.path.basename(__file__), '')
ICON_PATH               = os.path.join(PROG_PATH, "media", "icons")
PROGRAM_ICON_PATH       = os.path.join(ICON_PATH, "blink-monitor.png")

DATABASE_NAME = 'wellness_at_work_db'
TABLE_NAME = 'user1'
DATABASE_DATA_LOGGING_INTERVAL = 1
DATA_STREAMING_INTERVAL = 1
DATA_MONITORING_INTERVAL = 1
PERFORMANCE_LOGGING_INTERVAL = 1

MAX_FRAME_BUFFER_SIZE = FPS
MAX_FRAME_QUEUE_SIZE = 4
frame_que = [[[None]*2]*MAX_FRAME_BUFFER_SIZE]*MAX_FRAME_QUEUE_SIZE
current_frame_buffer_index = -1
current_frame_queue_index = 0

RECENT_DATA_COUNT = 10
recent_blink_count_list = [0]*RECENT_DATA_COUNT
recent_time_stamp_list = [int(datetime.now().timestamp())]*RECENT_DATA_COUNT

blink_monitor_logger = logging.getLogger("BlinkMonitor")
blink_monitor_logger.setLevel(logging.DEBUG)

blink_monitor_logger_channel = logging.StreamHandler()
blink_monitor_logger_channel.setLevel(logging.DEBUG) 

blink_monitor_logger.addHandler(blink_monitor_logger_channel)

blink_monitor_logger_file_handler = logging.FileHandler('Blink-Monitor.log')
blink_monitor_logger.addHandler(blink_monitor_logger_file_handler)


LOG_COLOR_LIST = {
    "DEBUG": "magenta", 
    "INFO": "",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red"    
}

class BlinkMonitor:
    def __init__(self):
        self.analyzer = FaceAnalyzer()
        self.monitoring_blink = False
    
    def start_blink_monitoring(self):
        blink_monitor_logger.info("Blink Monitor (Processing) Started")
        self.monitoring_blink = True
        while self.monitoring_blink:
            processed_frame, metrics = self.analyzer.process_frame(frame_que[current_frame_queue_index][current_frame_buffer_index-1][1])
   
    def stop_blink_monitoring(self):
        self.monitoring_blink = False
        blink_monitor_logger.info("Blink Monitor (Processing) Stopped")


    def blink_count(self):
        return self.analyzer.blink_counter


class FrameFetcher:
    def __init__(self):
        global current_frame_que_index
        self.camera_index = find_working_camera()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        self.fetching_frame = False

        if not self.cap.isOpened():
            blink_monitor_logger.error("Error: Camera not accessible")

    def start_frame_fetching(self):
        global frame_que, current_frame_buffer_index, current_frame_queue_index
        self.fetching_frame = True
        blink_monitor_logger.info("Frame Fetching (Capturing) Started")
        while self.fetching_frame:
            start_time = time.perf_counter() 
            for current_frame_queue_index in range(0, MAX_FRAME_QUEUE_SIZE):
                for current_frame_buffer_index in range(0, MAX_FRAME_BUFFER_SIZE):
                    frame_que[current_frame_queue_index][current_frame_buffer_index] = self.cap.read()
            end_time = time.perf_counter()
            blink_monitor_logger.info(f'{((MAX_FRAME_BUFFER_SIZE*MAX_FRAME_QUEUE_SIZE)/(end_time-start_time))} fps')

    def stop_frame_fetching(self):
        self.fetching_frame = False
        blink_monitor_logger.info("Frame Fetching (Capturing) Stopped")


class BlinkMonitorMain:
    def __init__(self):    
        self.frame_fetcher = FrameFetcher()
        self.blink_monitor = BlinkMonitor()
        self.performance_monitor = PerformnanceLogger()
        self.database_data_logger = DatabaseDataLogger()

        self.database_data_logging_thread = None

    def start_blink_monitoring(self):
        self.frame_fetching_thread = threading.Thread(target =self.frame_fetcher.start_frame_fetching)
        self.blink_monitoring_thread = threading.Thread(target =self.blink_monitor.start_blink_monitoring)
        self.performance_monitoring_thread = threading.Thread(target =self.performance_monitor.start_monitoring_performance)
        self.data_logging_thread = threading.Thread(target =self.database_data_logger.start_database_data_logging)
                
        self.frame_fetching_thread.start()
        self.blink_monitoring_thread.start()
        self.performance_monitoring_thread.start()
        self.data_logging_thread.start()

    def data_streaming(self):
        global recent_blink_count_list, recent_time_stamp_list
        blink_monitor_logger.info("Data Streaming Started")
        self.data_streaming_ongoing = True
        while self.data_streaming_ongoing:
            time.sleep(DATA_STREAMING_INTERVAL)
            recent_blink_count_list.insert(0, self.blink_monitor.blink_count())
            recent_time_stamp_list.insert(0, int(datetime.now().timestamp()))
            recent_blink_count_list.pop(-1)
            recent_time_stamp_list.pop(-1)

    def start_data_streaming(self):
        self.data_streaming_thread = threading.Thread(target =self.data_streaming)
        self.data_streaming_thread.start()

    def stop_data_streaming(self):
        self.data_streaming_ongoing = False
        self.data_streaming_thread.join()
        blink_monitor_logger.info("Data Streaming Stopped")


    def stop_blink_monitoring(self):
        self.database_data_logger.stop_database_data_logging()
        self.stop_data_streaming()
        self.blink_monitor.stop_blink_monitoring()
        self.frame_fetcher.stop_frame_fetching()
        self.performance_monitor.stop_performance_monitoring()
        blink_monitor_logger.info("Blink Monitor Main Stopped")


class DatabaseDataLogger:
    def __init__(self):
        self.database_data_logger_logging = False



    def database_data_logging(self):
        self.database_data_logger_logging = True
        self.database_connection = sqlite3.connect(DATABASE_NAME)
        self.connection_cursor = self.database_connection.cursor()

        self.connection_cursor.execute('''
        CREATE TABLE IF NOT EXISTS user1 (
            timestamp INTEGER NOT NULL,   
            blinkcount INTEGER NOT NULL
        )
        ''')
        self.database_connection.commit()

        while self.database_data_logger_logging:
            time.sleep(DATABASE_DATA_LOGGING_INTERVAL)
            self.log_data(int(datetime.now().timestamp()), recent_blink_count_list[-1])
        self.database_connection.close()

    def start_database_data_logging(self):
        self.database_data_logging_thread = threading.Thread(target = self.database_data_logging)
        self.database_data_logging_thread.start()
        blink_monitor_logger.info("Database Data logging Started")


    def stop_database_data_logging(self):
        self.database_data_logger_logging = False
        if self.database_data_logging_thread:
            self.database_data_logging_thread.join()
        blink_monitor_logger.info("Database Data logging Stopped")


    def log_data(
        self, 
        time_stamp, 
        blink_count
    ):
        self.connection_cursor.execute(f"INSERT INTO {TABLE_NAME} (timestamp, blinkcount) VALUES ({time_stamp}, {blink_count})")
        self.database_connection.commit()

'''    
class ProgramLogger:

'''

class PerformnanceLogger:
    def __init__(self, parent=None):
        self.process_id = psutil.Process()
        self.monitoring_performance = False

    def monitor_performance(self):
        self.monitoring_performance = True
        while self.monitoring_performance:
            cpu_usage = self.process_id.cpu_percent(interval=PERFORMANCE_LOGGING_INTERVAL)  
            ram_usage = self.process_id.memory_info().rss 

            ram_usage_mb = ram_usage / (1024 ** 2)
            blink_monitor_logger.info(f"CPU Usage: {cpu_usage}% | RAM Usage: {ram_usage_mb:.2f} MB")
            time.sleep(PERFORMANCE_LOGGING_INTERVAL) 

    def start_monitoring_performance(self):
        self.performance_monitoring_thread = threading.Thread(target = self.monitor_performance)
        self.performance_monitoring_thread.start()
        blink_monitor_logger.info("Performance Monitorng Started")

    def stop_performance_monitoring(self):
        self.monitoring_performance  = False
        self.performance_monitoring_thread.join()


class BlinkMonitorWindow(QMainWindow):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowIcon(QIcon(ICON_PATH))

        self.central_widget = QWidget()
        self.central_widget_layout = QVBoxLayout()

        self.central_widget.setLayout(self.central_widget_layout)
        self.setCentralWidget(self.central_widget)
    
        blink_monitor_logger.handlers[0].emit = self.handle_log_stream

        self.init_plot_view()
        self.init_display()
        self.init_log_view()
        self.init_control_pane()

        self.blink_monitor_main = BlinkMonitorMain()
        self.blink_monitor_main.start_blink_monitoring()
        self.blink_monitor_main.start_data_streaming()

        self.start_graphical_monitoring()

    def init_control_pane(self):
        self.control_pane_layout = QHBoxLayout()
        self.start_graphical_monitoring_button = QPushButton("Start Graphical Monitoring")
        self.stop_graphical_monitoring_button = QPushButton("Stop Graphical Monitoring")

        self.start_graphical_monitoring_button.clicked.connect(self.start_graphical_monitoring)
        self.stop_graphical_monitoring_button.clicked.connect(self.stop_graphical_monitoring)
        
        self.control_pane_layout.addWidget(self.start_graphical_monitoring_button)
        self.control_pane_layout.addWidget(self.stop_graphical_monitoring_button)

        self.central_widget_layout.addLayout(self.control_pane_layout)
        
    def init_plot_view(self):
        self.figure_handle = plotter.figure()
        self.plotting_axis_handle = self.figure_handle.add_subplot(111)
        self.canvas = FigureCanvas(self.figure_handle)
        self.central_widget_layout.addWidget(self.canvas)
        self.blink_monitor_plot_line = None

        self.plotting_axis_handle.set_xlabel("Time (s)",           fontsize=16, fontweight="bold")
        self.plotting_axis_handle.set_ylabel("Blinks Count",       fontsize=16, fontweight="bold")
        self.plotting_axis_handle.set_title("Blink Monitoring",    fontsize=16, fontweight="bold")
        
    def init_log_view(self):
        self.log_viewer = QTextBrowser()
        self.central_widget_layout.addWidget(self.log_viewer)
        
    def init_display(self):
        self.display_pane_layout = QVBoxLayout()
        self.total_blink_count_label = QLabel("Total Blink Count")
        self.total_blink_count_entry = QLabel("0")
        self.display_pane_layout.addWidget(self.total_blink_count_entry)
        self.display_pane_layout.addWidget(self.total_blink_count_label)

        self.central_widget_layout.addLayout(self.display_pane_layout)  

        self.total_blink_count_entry_update_timer = QTimer()
        self.total_blink_count_entry_update_timer.timeout.connect(self.update_blink_count_entry)

    def update_blink_count_entry(self):
        self.total_blink_count_entry.setText(str(recent_blink_count_list[0]))


    def update_plot(self):
        if self.blink_monitor_plot_line is not None:
            self.blink_monitor_plot_line.pop(0).remove()
        
        self.blink_monitor_plot_line = self.plotting_axis_handle.plot(
            recent_time_stamp_list,
            recent_blink_count_list,
            color='blue',
            label='Total Blinks'
        )
        plotter.legend(loc = 'upper right')
        self.plotting_axis_handle.set_xlim(recent_time_stamp_list[0], recent_time_stamp_list[-1])
        self.figure_handle.tight_layout()
        self.canvas.draw()  


    def graphical_monitoring(self):
        self.graphical_monitoring_ongoing = True
        while self.graphical_monitoring_ongoing:
            time.sleep(DATA_MONITORING_INTERVAL)
            self.update_plot()

    def start_graphical_monitoring(self):
        self.graphical_monitoring_thread = threading.Thread(target = self.graphical_monitoring)
        self.graphical_monitoring_thread.start()
        blink_monitor_logger.info("Graphical Monitorng Started")
        self.total_blink_count_entry_update_timer.start(DATA_MONITORING_INTERVAL*1000)

    def stop_graphical_monitoring(self):
        self.graphical_monitoring_ongoing = False
        self.total_blink_count_entry_update_timer.stop()
        self.graphical_monitoring_thread.join()
        blink_monitor_logger.info("Graphical Monitorng Stopped")


    def closeEvent(self, event):
        self.stop_graphical_monitoring()
        self.blink_monitor_main.stop_data_streaming()
        self.blink_monitor_main.stop_blink_monitoring()
        
        event.accept()

    def handle_log_stream(self, record):
        log_message = f'<span style="color: blue;"><b>{record.module}:</b></span> [{record.levelname}] <span style="color: {LOG_COLOR_LIST[record.levelname]};">{record.getMessage()}</span>'       
        self.log_viewer.append(log_message)


def main():
    blink_monitor_application = QApplication(sys.argv)
    blink_monitor_application.setWindowIcon(QIcon(PROGRAM_ICON_PATH))

    splash_screen = QSplashScreen(QPixmap(PROGRAM_ICON_PATH))
    splash_screen.show()

    blink_monitor_application_window = BlinkMonitorWindow() # Most time is consumed here.
    blink_monitor_application_window.setWindowTitle("Blink Monitor")
    splash_screen.finish(blink_monitor_application_window)

    blink_monitor_application_window.show()    
    sys.exit(blink_monitor_application.exec_())




if __name__ == "__main__":
    main()