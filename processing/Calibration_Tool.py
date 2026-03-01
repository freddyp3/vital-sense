import sys
import json
import serial
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal

SERIAL_PORT = 'COM4'   
BAUD_RATE = 921600
FS = 100               
WINDOW_SIZE = FS * 3   # Only a 3-second sliding window for instant feedback

class CSI_Reader_Thread(QThread):
    data_ready = pyqtSignal(np.ndarray)

    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.is_running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
        except Exception as e:
            print(f"Error: {e}")
            return

        while self.is_running:
            try:
                raw_line = ser.readline().decode('utf-8').strip()
                if raw_line.startswith("CSI_DATA"):
                    start_idx = raw_line.find("[")
                    end_idx = raw_line.find("]") + 1 
                    if start_idx != -1 and end_idx != 0:
                        csi_raw_data = json.loads(raw_line[start_idx:end_idx])
                        I = np.array(csi_raw_data[1::2])
                        Q = np.array(csi_raw_data[0::2])
                        
                        # CALIBRATION SECRET: Calculate Amplitude, not Phase!
                        packet_amp = np.abs(I + 1j * Q)
                        self.data_ready.emit(packet_amp)
            except: pass
        ser.close()

    def stop(self):
        self.is_running = False
        self.wait()

class CalibrationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antenna Layout Calibrator (Amplitude Mode)")
        self.resize(1000, 600)

        self.subcarriers = 0 
        self.time_array = np.linspace(-3, 0, WINDOW_SIZE)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Dashboard
        control_panel = QtWidgets.QGroupBox("Live Metrics")
        control_panel.setFixedWidth(300)
        form_layout = QtWidgets.QFormLayout(control_panel)

        self.strength_label = QtWidgets.QLabel("0")
        self.strength_label.setFont(QtGui.QFont("Arial", 28, QtGui.QFont.Bold))
        self.strength_label.setStyleSheet("color: #00ffcc;")
        
        self.variance_label = QtWidgets.QLabel("0")
        self.variance_label.setFont(QtGui.QFont("Arial", 28, QtGui.QFont.Bold))
        self.variance_label.setStyleSheet("color: #ffaa00;")

        form_layout.addRow(QtWidgets.QLabel("1. Baseline Strength (Higher is better):"))
        form_layout.addRow(self.strength_label)
        form_layout.addRow(QtWidgets.QLabel("2. Movement Energy (Spikes on movement):"))
        form_layout.addRow(self.variance_label)
        main_layout.addWidget(control_panel)

        # Plots
        plot_layout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(plot_layout)

        # Top Plot: Raw Amplitude
        self.amp_plot = plot_layout.addPlot(title="Raw Amplitude (Center Subcarrier)")
        self.amp_plot.setLabel('bottom', 'Time', units='s')
        self.amp_plot.setLabel('left', 'Wi-Fi Energy')
        self.amp_line = self.amp_plot.plot(pen=pg.mkPen(color='#00ffcc', width=2))
        
        plot_layout.nextRow()

        # Bottom Plot: Movement Variance
        self.var_plot = plot_layout.addPlot(title="Movement Sensitivity (Variance)")
        self.var_plot.setLabel('bottom', 'Time', units='s')
        self.var_plot.setLabel('left', 'Energy Change')
        self.var_line = self.var_plot.plot(pen=pg.mkPen(color='#ffaa00', width=2))

        self.amp_buffer = None
        self.var_buffer = np.zeros(WINDOW_SIZE)
        self.packet_count = 0

        self.serial_thread = CSI_Reader_Thread(SERIAL_PORT, BAUD_RATE)
        self.serial_thread.data_ready.connect(self.ingest_packet)
        self.serial_thread.start()

    def ingest_packet(self, packet_amp):
        if self.subcarriers == 0:
            self.subcarriers = len(packet_amp)
            self.amp_buffer = np.zeros((WINDOW_SIZE, self.subcarriers))
            self.center_idx = self.subcarriers // 2

        if len(packet_amp) != self.subcarriers: return

        self.amp_buffer[self.packet_count % WINDOW_SIZE, :] = packet_amp
        
        # Calculate Variance (Movement) over the last 0.5 seconds (50 samples)
        if self.packet_count > 50:
            idx = self.packet_count % WINDOW_SIZE
            recent_data = np.take(self.amp_buffer, range(idx-50, idx), mode='wrap', axis=0)
            current_var = np.var(recent_data[:, self.center_idx])
        else:
            current_var = 0

        self.var_buffer[self.packet_count % WINDOW_SIZE] = current_var
        self.packet_count += 1

        if self.packet_count % 10 == 0:
            self.update_ui()

    def update_ui(self):
        idx = self.packet_count % WINDOW_SIZE
        
        # Unroll buffers
        ordered_amp = np.concatenate([self.amp_buffer[idx:, self.center_idx], self.amp_buffer[:idx, self.center_idx]])
        ordered_var = np.concatenate([self.var_buffer[idx:], self.var_buffer[:idx]])

        # Update Graphs
        self.amp_line.setData(self.time_array, ordered_amp)
        self.var_line.setData(self.time_array, ordered_var)

        # Update Labels
        current_strength = ordered_amp[-1]
        self.strength_label.setText(f"{current_strength:.0f}")
        self.variance_label.setText(f"{ordered_var[-1]:.0f}")

    def closeEvent(self, event):
        self.serial_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CalibrationApp()
    window.show()
    sys.exit(app.exec_())

    