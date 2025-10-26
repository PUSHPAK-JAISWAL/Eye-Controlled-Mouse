
# 👁️ EyeController: Smoothed Eye-Gaze & Blink-Controlled Mouse

This repository contains the Python script for a hands-free mouse control system that uses a standard webcam, **MediaPipe Face Mesh**, and **OpenCV** to track eye movements and blinks. It features a separate **PyQt5** process for a large, visible cursor overlay and implements **Exponential Moving Average (EMA) smoothing** to minimize cursor jitter and improve usability.

## ✨ Features

* **Hands-Free Control:** Control the system cursor position using head movement (gaze tracking).
* **Jitter Suppression:** Implements an **EMA smoothing filter** to provide stable, predictable mouse movement.
* **Blink-to-Click:** Single-eye blinks are translated into Left and Right mouse clicks.
    * One eye blink (Left/Right) = Mouse Click (Left/Right, configurable).
    * Two-eye blink (Both) = Disabled (prevents accidental double-clicks).
* **Real-Time Calibration:** Quick, in-app calibration to correctly map which eye corresponds to which click action.
* **Visual Overlay:** A large, persistent green circle follows the cursor, running in a separate, dedicated PyQt5 process for reliability and visual feedback.
* **Accessibility Announcer:** Uses `pyttsx3` to announce when the cursor enters predefined regions (e.g., "Start Button", "Menu") or hovers over a new window (Windows only).

## 🛠️ Installation

### Prerequisites

* A webcam.
* A compatible Python environment (Python **3.8 to 3.11** is required due to MediaPipe constraints).

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd EyeController
    ```

2.  **Create and Activate a Virtual Environment:** (Using `uv` is recommended for speed)
    ```bash
    uv venv
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Run the Application

Execute the main script:

```bash
python EyeController_smoothed.py
````

A webcam feed window will open, and a large green cursor overlay will appear on your screen.

### 2\. Initial Setup & Controls

| Keypress | Action | Description |
| :--- | :--- | :--- |
| **L** | **Calibrate Left** | Blink your right eye once or twice to determine its blink ratio and establish normal/swapped mapping. |
| **R** | **Calibrate Right** | Blink your left eye once or twice to determine its blink ratio and establish normal/swapped mapping. |
| **M** | **Toggle Mapping** | Manually swap the mapping (Left Eye -\> Right Click / Right Eye -\> Left Click). |
| **F7** | **Toggle Mouse** | Enable or disable mouse cursor movement based on head position (stops or starts movement). |
| **Q** | **Quit (Window)** | Quit the application from the OpenCV window. |
| **Ctrl+Shift+Q** | **Global Quit** | A global hotkey to immediately shut down the main script and the cursor overlay process. |

-----

## 📦 Building a Standalone Executable

You can package this application into a single, distributable Windows executable using **PyInstaller**. This requires PyInstaller to be installed in your active environment: `uv pip install pyinstaller`.

1.  **Run the PyInstaller command:** (Using PowerShell syntax)

    ```powershell
    pyinstaller main.py `
    --name EyeController `
    --hidden-import win32api `
    --hidden-import pyttsx3.drivers `
    --hidden-import pyttsx3.drivers.sapi5 `
    --noconsole `
    --add-data "E:\Eye_Mouse\.venv\Lib\site-packages\mediapipe\modules;mediapipe\modules"
    ```

    *(**Note:** Adjust the `source_path` in the `--add-data` flag to match your specific environment path.)*

2.  **Distribute:** The final application package will be located in the `dist/EyeController` folder. You must distribute **the entire folder**.

## ⚙️ Configuration & Tuning

Key configuration variables are located near the top of `EyeController_smoothed.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SMOOTH_ALPHA` | `0.18` | **EMA Smoothing factor.** Lower for more delay/smoothness, higher for snappier response (Range: 0.0 - 1.0). |
| `MOVE_THRESHOLD` | `3` | Minimum movement (in pixels) before the filtered target moves the actual system mouse. |
| `BLINK_RATIO_THRESHOLD`| `0.22` | Eye Aspect Ratio threshold below which an eye is considered closed. |
| `BLINK_DEBOUNCE_SEC` | `0.45` | Time (in seconds) to wait after a click before another click can register. |

-----

## 🛑 Known Issues

  * **MediaPipe $3.12+$:** MediaPipe does not provide pre-compiled binaries for Python $3.12$ or newer on Windows. Users **must** use Python $3.11$ or earlier.
  * **Lighting:** Reliable eye-gaze tracking requires consistent, non-direct lighting. Sunlight or harsh backlighting can degrade performance.

```
```

