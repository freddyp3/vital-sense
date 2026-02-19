# Vital-Sense Radar

Non-contact respiratory and heart rate monitoring using ESP32-S3 Wi-Fi CSI and Python signal processing.

---

## Project Structure

```text
vital-sense/
├── firmware/        # ESP-IDF firmware (esp-csi lives here)
├── processing/      # Python logging + signal processing
└── vital_env/       # Python virtual environment (DO NOT COMMIT)
```

---

# 1) Firmware Setup (ESP32-S3)

This section builds and flashes CSI firmware onto ESP32-S3 boards.

---

## Prerequisites

Install:

- ESP-IDF v5.5.x
- ESP-IDF Tools (CMake, Ninja, Python managed automatically)

Official install guide:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/

---

## macOS / Linux

### Activate ESP-IDF (Required Per New Terminal)

```bash
. ~/.espressif/tools/activate_idf_v5.5.3.sh
```

---

### Build CSI Sender

```bash
cd firmware/esp-csi/examples/get-started/csi_send
idf.py set-target esp32s3
idf.py build
```

---

### Build CSI Receiver

```bash
cd firmware/esp-csi/examples/get-started/csi_recv
idf.py set-target esp32s3
idf.py build
```

---

### Flash + Monitor (Only When Board Connected)

Find serial port:

```bash
ls /dev/tty.*
```

Flash:

```bash
idf.py -p /dev/tty.YOUR_PORT flash monitor
```

Exit monitor:

```
Ctrl + ]
```

---

## Windows (PowerShell)

**Recommended:** Use “ESP-IDF Command Prompt” installed with ESP-IDF.

If using PowerShell directly:

### Activate ESP-IDF

```powershell
& "$HOME\.espressif\tools\activate_idf_v5.5.3.ps1"
```

---

### Build CSI Sender

```powershell
cd firmware\esp-csi\examples\get-started\csi_send
idf.py set-target esp32s3
idf.py build
```

---

### Build CSI Receiver

```powershell
cd firmware\esp-csi\examples\get-started\csi_recv
idf.py set-target esp32s3
idf.py build
```

---

### Flash + Monitor

Check COM port in Device Manager.

```powershell
idf.py -p COM4 flash monitor
```

Exit monitor:

```
Ctrl + ]
```

---

# 2) Processing Setup (Python Signal Processing)

This runs the CSI logging and signal processing pipeline.

This is separate from ESP-IDF.

---

## Create Virtual Environment (All OS)

From project root:

```bash
python -m venv vital_env
```

---

### Activate (macOS / Linux)

```bash
source vital_env/bin/activate
```

---

### Activate (Windows PowerShell)

```powershell
.\vital_env\Scripts\Activate.ps1
```

---

## Install Dependencies

Using requirements file:

```bash
pip install -r processing/requirements.txt
```

Or manually:

```bash
pip install numpy scipy pyserial matplotlib
```

---

## Run Logger

```bash
cd processing
python logger.py
```

---

# 3) Important Notes

### Firmware Environment

- You must activate ESP-IDF **every time you open a new terminal**
- `idf.py` will not work without activation
- `set-target esp32s3` only needs to be run once per project unless `sdkconfig` is deleted

### Python Environment

- `vital_env/` is local to each developer machine
- Do NOT commit `vital_env/`
- Activate the venv before running logger

---

# 4) Recommended .gitignore

Add this to your `.gitignore`:

```text
# Python
vital_env/
__pycache__/
*.pyc

# ESP-IDF
build/
sdkconfig
sdkconfig.old

# macOS
.DS_Store
```

---

# 5) Common Errors

### "command not found: idf.py"
You did not activate ESP-IDF in that terminal session.

### "IDF_TARGET mismatch"
Run:
```bash
idf.py set-target esp32s3
```

### Python using wrong version
Check:
```bash
python --version
```
Make sure virtual environment is activated.

---

# 6) High-Level Workflow

Terminal 1 (Firmware TX):
- Activate ESP-IDF
- Build
- Flash

Terminal 2 (Firmware RX):
- Activate ESP-IDF
- Build
- Flash

Terminal 3 (Processing):
- Activate virtual environment
- Run logger

---

Project maintained for multi-OS collaboration (macOS, Linux, Windows).