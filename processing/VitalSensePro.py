import sys
import json
import serial
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.signal import butter, filtfilt, medfilt, find_peaks, detrend
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
SERIAL_PORT = 'COM4'   # Double check this is still correct!
BAUD_RATE = 921600
FS = 100               # 100 packets per second
WINDOW_SECONDS = 15    
WINDOW_SIZE = FS * WINDOW_SECONDS  

# --- SIGNAL PROCESSING ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if lowcut >= highcut: return data * 0 
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ==============================================================================
# 1. THE BACKGROUND WORKER THREAD (Bulletproof Serial Reading)
# ==============================================================================
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
            print(f"Hardware Thread: Connected to {self.port}")
        except Exception as e:
            print(f"Hardware Thread Error: Could not open {self.port}. {e}")
            return

        while self.is_running:
            try:
                raw_line = ser.readline().decode('utf-8').strip()
                
                if raw_line.startswith("CSI_DATA"):
                    start_idx = raw_line.find("[")
                    end_idx = raw_line.find("]") + 1 
                    
                    if start_idx != -1 and end_idx != 0:
                        data_string = raw_line[start_idx:end_idx]
                        csi_raw_data = json.loads(data_string)
                        
                        I = np.array(csi_raw_data[1::2])
                        Q = np.array(csi_raw_data[0::2])
                        
                        complex_csi = I + 1j * Q
                        packet_phase = np.angle(complex_csi)
                        
                        self.data_ready.emit(packet_phase)

            except json.JSONDecodeError:
                pass 
            except serial.SerialException as e:
                print(f"Hardware Thread: Serial disconnected — {e}")
                self.is_running = False
                break
            except Exception:
                pass

        ser.close()

    def stop(self):
        self.is_running = False
        self.wait()

# ==============================================================================
# 2. THE MAIN GUI APPLICATION
# ==============================================================================
class RadarApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vital-Sense Pro V5.1: Unlocked Y-Axis")
        self.resize(1200, 750)

        self.subcarriers = 0 
        self.time_array = np.linspace(-WINDOW_SECONDS, 0, WINDOW_SIZE)

        # --- BUILD THE UI ---
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Dashboard Panel
        control_panel = QtWidgets.QGroupBox("Radar Dashboard")
        control_panel.setFixedWidth(320)
        form_layout = QtWidgets.QFormLayout(control_panel)

        self.bpm_label = QtWidgets.QLabel("-- Br/min")
        self.bpm_label.setFont(QtGui.QFont("Arial", 36, QtGui.QFont.Bold))
        self.bpm_label.setStyleSheet("color: #00ffcc; margin-bottom: 10px;")
        self.bpm_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.status_label = QtWidgets.QLabel("Status: Waiting for data...")
        self.status_label.setStyleSheet("color: yellow; font-weight: bold; margin-bottom: 10px;")

        # DSP Tuning
        self.lowcut_spin = QtWidgets.QDoubleSpinBox(); self.lowcut_spin.setRange(0.01, 50.0); self.lowcut_spin.setSingleStep(0.05); self.lowcut_spin.setValue(0.20)
        self.highcut_spin = QtWidgets.QDoubleSpinBox(); self.highcut_spin.setRange(0.05, 50.0); self.highcut_spin.setSingleStep(0.05); self.highcut_spin.setValue(0.50)
        self.order_spin = QtWidgets.QSpinBox(); self.order_spin.setRange(1, 20); self.order_spin.setValue(4)
        self.kernel_spin = QtWidgets.QSpinBox(); self.kernel_spin.setRange(3, 999); self.kernel_spin.setSingleStep(2); self.kernel_spin.setValue(101)
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Filtered Data (Breathing)", "Raw PC1 Data (Debug)"])

        # Graph Scale Controls (Unlocked to +/- 10 Million)
        self.ymin_spin = QtWidgets.QDoubleSpinBox(); self.ymin_spin.setRange(-10000000.0, 10000000.0); self.ymin_spin.setSingleStep(50.0); self.ymin_spin.setValue(-50.0)
        self.ymax_spin = QtWidgets.QDoubleSpinBox(); self.ymax_spin.setRange(-10000000.0, 10000000.0); self.ymax_spin.setSingleStep(50.0); self.ymax_spin.setValue(100.0)

        self.reset_btn = QtWidgets.QPushButton("FLUSH BUFFER (Reset)")
        self.reset_btn.setStyleSheet("background-color: #aa0000; color: white; font-weight: bold; padding: 10px; margin-top: 10px;")
        self.reset_btn.clicked.connect(self.flush_buffer)

        form_layout.addRow(self.bpm_label)
        form_layout.addRow(self.status_label)
        form_layout.addRow("Lowcut (Hz):", self.lowcut_spin)
        form_layout.addRow("Highcut (Hz):", self.highcut_spin)
        form_layout.addRow("Filter Order:", self.order_spin)
        form_layout.addRow("Spike Window:", self.kernel_spin)
        form_layout.addRow("View Mode:", self.view_combo)
        form_layout.addRow(QtWidgets.QLabel("")) # Spacer
        form_layout.addRow("Y-Axis Min:", self.ymin_spin)
        form_layout.addRow("Y-Axis Max:", self.ymax_spin)
        form_layout.addRow(self.reset_btn)
        main_layout.addWidget(control_panel)

        # Plot Panel
        self.graph_widget = pg.PlotWidget(title="Live CSI Waveform")
        self.graph_widget.setBackground('#1e1e2e') 
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.graph_widget.setLabel('bottom', 'Time', units='Seconds')
        self.graph_widget.setLabel('left', 'Amplitude', units='Phase Variance')
        self.graph_widget.setXRange(-WINDOW_SECONDS, 0) # Lock X-Axis
        
        self.plot_line = self.graph_widget.plot(pen=pg.mkPen(color='#00ffcc', width=2.5))
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        self.graph_widget.addItem(self.peak_scatter)
        main_layout.addWidget(self.graph_widget)

        self.phase_buffer = None 
        self.is_buffer_full = False
        self.packet_count = 0

        # --- START THE HARDWARE THREAD ---
        self.serial_thread = CSI_Reader_Thread(SERIAL_PORT, BAUD_RATE)
        self.serial_thread.data_ready.connect(self.ingest_packet)
        self.serial_thread.start()

    def flush_buffer(self):
        if self.subcarriers > 0:
            self.phase_buffer = np.zeros((WINDOW_SIZE, self.subcarriers))
        self.is_buffer_full = False
        self.packet_count = 0
        self.bpm_label.setText("Wait 15s...")
        self.plot_line.setData([]) 
        self.peak_scatter.setData([]) 
        print("Buffer flushed.")

    def ingest_packet(self, packet_phase):
        # 1. LOCK THE SUBCARRIERS ON THE FIRST PACKET
        if self.subcarriers == 0:
            print(f"Hardware locked in at {len(packet_phase)} subcarriers.")
            self.subcarriers = len(packet_phase)
            self.flush_buffer()

        # 2. IGNORE CORRUPTED/ANOMALOUS PACKETS (Do not reset!)
        if len(packet_phase) != self.subcarriers:
            return

        # 3. Insert valid data into the buffer
        self.phase_buffer[self.packet_count % WINDOW_SIZE, :] = packet_phase
        self.packet_count += 1

        if not self.is_buffer_full:
            percent = int((self.packet_count / WINDOW_SIZE) * 100)
            self.status_label.setText(f"Status: Calibrating ({percent}%)")
            
        if self.packet_count >= WINDOW_SIZE:
            if not self.is_buffer_full:
                self.status_label.setText("Status: LIVE")
                self.status_label.setStyleSheet("color: #00ffcc; font-weight: bold;")
            self.is_buffer_full = True

            if self.packet_count % 10 == 0:
                self.run_signal_processing()

    def run_signal_processing(self):
        # 1. Permanently lock the Y-Axis to the values in the text boxes
        self.graph_widget.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.graph_widget.setYRange(self.ymin_spin.value(), self.ymax_spin.value(), padding=0)

        # 2. DSP Pipeline
        idx = self.packet_count % WINDOW_SIZE
        ordered = np.concatenate([self.phase_buffer[idx:], self.phase_buffer[:idx]], axis=0)
        unwrapped = np.unwrap(ordered, axis=0)
        
        # THE MAGIC FIX: Flatten the infinite clock drift slope
        detrended = detrend(unwrapped, axis=0)
        
        pc1 = PCA(n_components=1).fit_transform(detrended)[:, 0]
        
        kernel = self.kernel_spin.value()
        if kernel % 2 == 0: kernel += 1 
        clean_pc1 = medfilt(pc1, kernel)

        mode = self.view_combo.currentText()
        if mode == "Filtered Data (Breathing)":
            output_wave = butter_bandpass_filter(clean_pc1, self.lowcut_spin.value(), self.highcut_spin.value(), FS, self.order_spin.value())
            
            peaks, _ = find_peaks(output_wave, distance=FS*1.5)
            if len(peaks) >= 2:
                avg_distance_sec = np.mean(np.diff(peaks)) / FS
                bpm = 60.0 / avg_distance_sec
                self.bpm_label.setText(f"{bpm:.1f} Br/min")
                self.peak_scatter.setData(self.time_array[peaks], output_wave[peaks])
            else:
                self.bpm_label.setText("-- Br/min")
                self.peak_scatter.setData([])

        else:
            output_wave = clean_pc1 
            self.peak_scatter.setData([])

        self.plot_line.setData(self.time_array, output_wave)

    def closeEvent(self, event):
        self.serial_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = pg.QtGui.QPalette()
    palette.setColor(pg.QtGui.QPalette.Window, pg.QtGui.QColor(40, 40, 40))
    palette.setColor(pg.QtGui.QPalette.WindowText, QtCore.Qt.white)
    app.setPalette(palette)
    window = RadarApp()
    window.show()
    sys.exit(app.exec_())