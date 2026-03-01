import sys
import json
import serial
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
SERIAL_PORT = 'COM4'   
BAUD_RATE = 921600
FS = 100               

WINDOW_SECONDS = 15    
WINDOW_SIZE = FS * WINDOW_SECONDS  
MAX_HIST_SECONDS = 300 
MAX_HIST_SIZE = FS * MAX_HIST_SECONDS

SQUELCH_SECONDS = 2.0  

# --- SIGNAL PROCESSING ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if lowcut >= highcut: return data * 0 
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ==============================================================================
# 1. THE BACKGROUND WORKER THREAD 
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
        except Exception as e:
            print(f"Hardware Thread Error: {e}")
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
                        packet_amp = np.abs(I + 1j * Q)
                        
                        self.data_ready.emit(packet_amp)
            except: pass
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
        self.setWindowTitle("Vital-Sense Pro V17: Dual Sync View & Flush Delay")
        self.resize(1350, 850)

        self.subcarriers = 0 
        self.is_flushing = False # Flag to ignore packets during the 3-second delay
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Dashboard Panel
        control_panel = QtWidgets.QGroupBox("Radar Dashboard")
        control_panel.setFixedWidth(360)
        form_layout = QtWidgets.QFormLayout(control_panel)

        self.bpm_label = QtWidgets.QLabel("-- Br/min")
        self.bpm_label.setFont(QtGui.QFont("Arial", 36, QtGui.QFont.Bold))
        self.bpm_label.setStyleSheet("color: #00ffcc; margin-bottom: 10px;")
        self.bpm_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.status_label = QtWidgets.QLabel("Status: Waiting for data...")
        self.status_label.setStyleSheet("color: yellow; font-weight: bold; margin-bottom: 10px;")

        # GATING CONTROLS
        self.mean_display = QtWidgets.QLabel("0.0")
        self.mean_display.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 14px;")
        
        self.var_display = QtWidgets.QLabel("0.0")
        self.var_display.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        
        self.presence_spin = QtWidgets.QDoubleSpinBox()
        self.presence_spin.setRange(0.0, 500.0)
        self.presence_spin.setSingleStep(1.0)
        self.presence_spin.setValue(45.0) # Requested Default
        
        self.motion_spin = QtWidgets.QDoubleSpinBox()
        self.motion_spin.setRange(1.0, 5000.0)
        self.motion_spin.setSingleStep(5.0)
        self.motion_spin.setValue(30.0) # Requested Default

        self.lowcut_spin = QtWidgets.QDoubleSpinBox(); self.lowcut_spin.setRange(0.01, 50.0); self.lowcut_spin.setSingleStep(0.05); self.lowcut_spin.setValue(0.20)
        self.highcut_spin = QtWidgets.QDoubleSpinBox(); self.highcut_spin.setRange(0.05, 50.0); self.highcut_spin.setSingleStep(0.05); self.highcut_spin.setValue(0.50)
        self.order_spin = QtWidgets.QSpinBox(); self.order_spin.setRange(1, 20); self.order_spin.setValue(4)
        self.kernel_spin = QtWidgets.QSpinBox(); self.kernel_spin.setRange(3, 999); self.kernel_spin.setSingleStep(2); self.kernel_spin.setValue(101)
        
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Dual View (Synchronized)", "Filtered Data (Breathing)", "Raw PC1 Data (Debug)"])

        # Time Controls
        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 5.0)
        self.delay_spin.setSingleStep(0.5)
        self.delay_spin.setValue(1.0) # Requested Default

        self.live_scroll_cb = QtWidgets.QCheckBox("Live Scroll (Follows Delay)")
        self.live_scroll_cb.setChecked(True)
        self.live_scroll_cb.stateChanged.connect(self.toggle_time_mode)
        
        self.xaxis_spin = QtWidgets.QSpinBox()
        self.xaxis_spin.setRange(5, MAX_HIST_SECONDS)
        self.xaxis_spin.setSingleStep(5)
        self.xaxis_spin.setValue(15) # Requested Default
        
        self.view_end_spin = QtWidgets.QDoubleSpinBox()
        self.view_end_spin.setRange(5, 999999)
        self.view_end_spin.setSingleStep(5)
        self.view_end_spin.setDecimals(1)
        self.view_end_spin.setEnabled(False) 

        # Scrubbing
        scrub_layout = QtWidgets.QHBoxLayout()
        self.btn_left = QtWidgets.QPushButton("<<<")
        self.btn_right = QtWidgets.QPushButton(">>>")
        self.scrub_step_spin = QtWidgets.QDoubleSpinBox()
        self.scrub_step_spin.setRange(1.0, 60.0)
        self.scrub_step_spin.setSingleStep(1.0)
        self.scrub_step_spin.setValue(5.0) 
        
        self.btn_left.clicked.connect(self.scroll_left)
        self.btn_right.clicked.connect(self.scroll_right)
        
        scrub_layout.addWidget(self.btn_left)
        scrub_layout.addWidget(self.scrub_step_spin)
        scrub_layout.addWidget(self.btn_right)

        self.autoscale_cb = QtWidgets.QCheckBox("Auto-Scale Y-Axis")
        self.autoscale_cb.setChecked(True)
        self.ymin_spin = QtWidgets.QDoubleSpinBox(); self.ymin_spin.setRange(-10000000.0, 10000000.0); self.ymin_spin.setSingleStep(50.0); self.ymin_spin.setValue(-10.0)
        self.ymax_spin = QtWidgets.QDoubleSpinBox(); self.ymax_spin.setRange(-10000000.0, 10000000.0); self.ymax_spin.setSingleStep(50.0); self.ymax_spin.setValue(10.0)

        self.reset_btn = QtWidgets.QPushButton("FLUSH BUFFER (3s Delay)")
        self.reset_btn.setStyleSheet("background-color: #aa0000; color: white; font-weight: bold; padding: 10px; margin-top: 10px;")
        self.reset_btn.clicked.connect(self.trigger_flush_delay)

        form_layout.addRow(self.bpm_label)
        form_layout.addRow(self.status_label)
        
        form_layout.addRow(QtWidgets.QLabel("--- SIGNAL GATING ---"))
        form_layout.addRow("Live Mean (Strength):", self.mean_display)
        form_layout.addRow("Live Variance (Motion):", self.var_display)
        form_layout.addRow("Empty Room Gate (Max Mean):", self.presence_spin)
        form_layout.addRow("Motion Gate (Max Variance):", self.motion_spin)
        
        form_layout.addRow(QtWidgets.QLabel("--- DSP TUNING ---"))
        form_layout.addRow("Lowcut (Hz):", self.lowcut_spin)
        form_layout.addRow("Highcut (Hz):", self.highcut_spin)
        form_layout.addRow("Filter Order:", self.order_spin)
        form_layout.addRow("Spike Window:", self.kernel_spin)
        form_layout.addRow("View Mode:", self.view_combo)
        
        form_layout.addRow(QtWidgets.QLabel("--- X-AXIS TIME CONTROLS ---"))
        form_layout.addRow("Live Filter Delay (s):", self.delay_spin)
        form_layout.addRow(self.live_scroll_cb)
        form_layout.addRow("Window Length (s):", self.xaxis_spin)
        form_layout.addRow("View End Time (s):", self.view_end_spin)
        form_layout.addRow("Time Scrubbing:", scrub_layout)
        
        form_layout.addRow(QtWidgets.QLabel("--- Y-AXIS SCALE CONTROLS ---"))
        form_layout.addRow(self.autoscale_cb)
        form_layout.addRow("Y-Axis Min:", self.ymin_spin)
        form_layout.addRow("Y-Axis Max:", self.ymax_spin)
        form_layout.addRow(self.reset_btn)
        main_layout.addWidget(control_panel)

        # --- DUAL GRAPH LAYOUT ---
        graph_layout = QtWidgets.QVBoxLayout()
        
        self.graph_filtered = pg.PlotWidget(title="Filtered Data (Breathing Waveform)")
        self.graph_filtered.setBackground('#1e1e2e') 
        self.graph_filtered.showGrid(x=True, y=True, alpha=0.3)
        self.graph_filtered.setLabel('left', 'Filtered Amp')
        self.plot_line_filtered = self.graph_filtered.plot(pen=pg.mkPen(color='#00ffcc', width=2.5))
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        self.graph_filtered.addItem(self.peak_scatter)
        
        self.graph_raw = pg.PlotWidget(title="Raw PC1 Data (Debug Waveform)")
        self.graph_raw.setBackground('#1e1e2e') 
        self.graph_raw.showGrid(x=True, y=True, alpha=0.3)
        self.graph_raw.setLabel('bottom', 'Absolute Time', units='Seconds')
        self.graph_raw.setLabel('left', 'Raw PC1 Amp')
        self.plot_line_raw = self.graph_raw.plot(pen=pg.mkPen(color='#ffaa00', width=1.5))
        
        # LINK THE X-AXES TOGETHER
        self.graph_filtered.setXLink(self.graph_raw)

        graph_layout.addWidget(self.graph_filtered)
        graph_layout.addWidget(self.graph_raw)
        main_layout.addLayout(graph_layout)

        self.amp_buffer = None 
        self.hist_pc1 = np.zeros(MAX_HIST_SIZE)
        self.hist_filtered = np.zeros(MAX_HIST_SIZE)
        self.hist_time = np.zeros(MAX_HIST_SIZE)
        
        self.last_pca_comp = None 
        self.packet_count = 0
        self.last_stitched_packet = 0 

        self.serial_thread = CSI_Reader_Thread(SERIAL_PORT, BAUD_RATE)
        self.serial_thread.data_ready.connect(self.ingest_packet)
        self.serial_thread.start()

    def toggle_time_mode(self):
        self.view_end_spin.setEnabled(not self.live_scroll_cb.isChecked())

    def scroll_left(self):
        self.live_scroll_cb.setChecked(False)
        step = self.scrub_step_spin.value()
        new_end = max(self.xaxis_spin.value(), self.view_end_spin.value() - step)
        self.view_end_spin.setValue(new_end)
        self.run_display_engine_only() 

    def scroll_right(self):
        self.live_scroll_cb.setChecked(False)
        step = self.scrub_step_spin.value()
        max_time = self.last_stitched_packet / FS
        new_end = min(max_time, self.view_end_spin.value() + step)
        self.view_end_spin.setValue(new_end)
        self.run_display_engine_only() 

    def trigger_flush_delay(self):
        self.is_flushing = True
        self.status_label.setText("Status: GET IN POSITION (3s...)")
        self.status_label.setStyleSheet("color: #ff00ff; font-weight: bold; font-size: 18px;")
        self.bpm_label.setText("-- Br/min")
        self.plot_line_filtered.setData([])
        self.plot_line_raw.setData([])
        self.peak_scatter.setData([])
        
        # Fire the actual flush exactly 3 seconds from now
        QtCore.QTimer.singleShot(3000, self.execute_flush)

    def execute_flush(self):
        if self.subcarriers > 0:
            self.amp_buffer = np.zeros((WINDOW_SIZE, self.subcarriers))
        self.hist_pc1 = np.zeros(MAX_HIST_SIZE)
        self.hist_filtered = np.zeros(MAX_HIST_SIZE)
        self.hist_time = np.zeros(MAX_HIST_SIZE)
        self.last_pca_comp = None
        self.packet_count = 0
        self.last_stitched_packet = 0
        self.status_label.setText("Status: Waiting for data...")
        self.status_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.is_flushing = False

    def ingest_packet(self, packet_amp):
        if self.is_flushing: 
            return # Ignore all packets while the user is getting into position!

        if self.subcarriers == 0:
            self.subcarriers = len(packet_amp)
            self.execute_flush()

        if len(packet_amp) != self.subcarriers: return

        self.amp_buffer[self.packet_count % WINDOW_SIZE, :] = packet_amp
        self.packet_count += 1

        if self.packet_count < WINDOW_SIZE:
            percent = int((self.packet_count / WINDOW_SIZE) * 100)
            self.status_label.setText(f"Status: Calibrating ({percent}%)")
            return

        update_interval = max(1, FS // 10)
        if self.packet_count % update_interval == 0:
            self.run_signal_processing()

    def run_signal_processing(self):
        idx = self.packet_count % WINDOW_SIZE
        ordered = np.concatenate([self.amp_buffer[idx:], self.amp_buffer[:idx]], axis=0)
        
        # SQUELCH LOGIC
        squelch_pkts = int(SQUELCH_SECONDS * FS)
        recent_data = ordered[-squelch_pkts:]
        
        current_mean = np.mean(recent_data)
        current_variance = np.mean(np.var(recent_data, axis=0))
        
        self.mean_display.setText(f"{current_mean:.1f}")
        self.var_display.setText(f"{current_variance:.1f}")

        presence_gate = self.presence_spin.value()
        motion_gate = self.motion_spin.value()

        if current_mean > presence_gate:
            self.status_label.setText("Status: EMPTY ROOM")
            self.status_label.setStyleSheet("color: #ff0000; font-weight: bold;")
            clean_pc1 = np.zeros(WINDOW_SIZE)
            filtered_wave = np.zeros(WINDOW_SIZE)
            
        elif current_variance > motion_gate:
            self.status_label.setText("Status: MOTION ARTIFACT / RINGING")
            self.status_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
            clean_pc1 = np.zeros(WINDOW_SIZE)
            filtered_wave = np.zeros(WINDOW_SIZE)
            
        else:
            self.status_label.setText("Status: LIVE TRACKING")
            self.status_label.setStyleSheet("color: #00ffcc; font-weight: bold;")
            
            centered = ordered - np.mean(ordered, axis=0)
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(centered)[:, 0]
            
            if self.last_pca_comp is not None:
                if np.dot(pca.components_[0], self.last_pca_comp) < 0:
                    pc1 = -pc1
                    pca.components_[0] = -pca.components_[0]
            self.last_pca_comp = pca.components_[0]
            
            kernel = self.kernel_spin.value()
            if kernel % 2 == 0: kernel += 1 
            clean_pc1 = medfilt(pc1, kernel)
            filtered_wave = butter_bandpass_filter(clean_pc1, self.lowcut_spin.value(), self.highcut_spin.value(), FS, self.order_spin.value())

        # OVERLAP-SAVE STITCHING
        delay_pkts = int(self.delay_spin.value() * FS)
        safe_boundary = self.packet_count - delay_pkts
        
        if safe_boundary > self.last_stitched_packet:
            n_new = safe_boundary - self.last_stitched_packet
            
            max_avail = WINDOW_SIZE - delay_pkts
            if n_new > max_avail:
                n_new = max_avail
                self.last_stitched_packet = safe_boundary - n_new 

            if delay_pkts > 0:
                stitch_pc1 = clean_pc1[-delay_pkts - n_new : -delay_pkts]
                stitch_filt = filtered_wave[-delay_pkts - n_new : -delay_pkts]
            else:
                stitch_pc1 = clean_pc1[-n_new :]
                stitch_filt = filtered_wave[-n_new :]

            self.hist_pc1 = np.roll(self.hist_pc1, -n_new)
            self.hist_pc1[-n_new:] = stitch_pc1
            
            self.hist_filtered = np.roll(self.hist_filtered, -n_new)
            self.hist_filtered[-n_new:] = stitch_filt
            
            start_t = self.last_stitched_packet / FS
            end_t = safe_boundary / FS
            new_times = np.linspace(start_t, end_t, n_new, endpoint=False)
            
            self.hist_time = np.roll(self.hist_time, -n_new)
            self.hist_time[-n_new:] = new_times

            self.last_stitched_packet = safe_boundary

        self.run_display_engine_only()

    def run_display_engine_only(self):
        window_len_sec = self.xaxis_spin.value()
        
        if self.live_scroll_cb.isChecked():
            end_time = self.last_stitched_packet / FS
            self.view_end_spin.setValue(end_time)
        else:
            end_time = self.view_end_spin.value()
            
        start_time = max(0, end_time - window_len_sec)

        # Apply bounds to both graphs
        self.graph_filtered.setXRange(start_time, end_time, padding=0)
        # Raw graph updates automatically because of setXLink
        
        if self.autoscale_cb.isChecked():
            self.graph_filtered.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.graph_raw.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        else:
            self.graph_filtered.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self.graph_filtered.setYRange(self.ymin_spin.value(), self.ymax_spin.value(), padding=0)
            self.graph_raw.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self.graph_raw.setYRange(self.ymin_spin.value(), self.ymax_spin.value(), padding=0)

        valid_start = max(0, MAX_HIST_SIZE - self.last_stitched_packet)
        valid_time = self.hist_time[valid_start:]
        valid_filtered = self.hist_filtered[valid_start:]
        valid_pc1 = self.hist_pc1[valid_start:]

        view_mask = (valid_time >= start_time) & (valid_time <= end_time)
        display_time = valid_time[view_mask]
        display_filtered = valid_filtered[view_mask]
        display_pc1 = valid_pc1[view_mask] 

        # Update the arrays in both graphs
        self.plot_line_filtered.setData(display_time, display_filtered)
        self.plot_line_raw.setData(display_time, display_pc1)

        # Handle View Modes Show/Hide
        mode = self.view_combo.currentText()
        if mode == "Dual View (Synchronized)":
            self.graph_filtered.show()
            self.graph_raw.show()
        elif mode == "Filtered Data (Breathing)":
            self.graph_filtered.show()
            self.graph_raw.hide()
        else:
            self.graph_filtered.hide()
            self.graph_raw.show()

        # Handle Peak Detection (only on filtered wave)
        if mode in ["Dual View (Synchronized)", "Filtered Data (Breathing)"]:
            if len(display_filtered) > 0 and not np.all(display_filtered == 0):
                peaks, _ = find_peaks(display_filtered, distance=FS*1.5)
                if len(peaks) >= 2:
                    recent_peaks = peaks[-5:] if len(peaks) >= 5 else peaks
                    avg_distance_sec = np.mean(np.diff(recent_peaks)) / FS
                    bpm = 60.0 / avg_distance_sec if avg_distance_sec > 0 else 0
                    self.bpm_label.setText(f"{bpm:.1f} Br/min")
                    self.peak_scatter.setData(display_time[peaks], display_filtered[peaks])
                else:
                    self.bpm_label.setText("-- Br/min")
                    self.peak_scatter.setData([])
            else:
                self.bpm_label.setText("-- Br/min")
                self.peak_scatter.setData([])
        else:
            self.bpm_label.setText("-- Br/min")
            self.peak_scatter.setData([])

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