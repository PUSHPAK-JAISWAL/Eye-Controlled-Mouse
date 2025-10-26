# EyeController_smoothed.py
# Combined script: original OpenCV MediaPipe eye-control UI + PyQt cursor overlay child process.
# Adds smoothing and jitter suppression for the system mouse movements.

import sys
import os
import time
import math
import cv2
import numpy as np
import pyautogui
import threading
import multiprocessing
import pyttsx3
import keyboard
import platform
from collections import deque

# ---------- CursorOverlay (runs in separate process) ----------
def run_cursor_overlay(radius=80, log_file="cursor_overlay_error.log"):
    try:
        # Import inside child process
        import sys as _sys
        import cv2 as _cv2
        import numpy as _np
        import pyautogui as _pyautogui
        from PyQt5 import QtWidgets as _QtWidgets, QtGui as _QtGui, QtCore as _QtCore

        class CursorOverlay(_QtWidgets.QWidget):
            def __init__(self, radius=30):
                super().__init__()
                self.radius = radius
                self.diameter = 2 * self.radius + 4
                self.setWindowFlags(
                    _QtCore.Qt.FramelessWindowHint |
                    _QtCore.Qt.WindowStaysOnTopHint |
                    _QtCore.Qt.Tool |
                    _QtCore.Qt.X11BypassWindowManagerHint
                )
                self.setAttribute(_QtCore.Qt.WA_TranslucentBackground)
                self.setAttribute(_QtCore.Qt.WA_NoSystemBackground)
                # Make overlay ignore mouse events so it does not intercept pointer/clicks
                self.setAttribute(_QtCore.Qt.WA_TransparentForMouseEvents)
                self.setFixedSize(self.diameter, self.diameter)

                self.label = _QtWidgets.QLabel(self)
                self.label.setGeometry(0, 0, self.diameter, self.diameter)

                self.timer = _QtCore.QTimer()
                self.timer.timeout.connect(self.update_position)
                self.timer.start(10)

            def update_position(self):
                x, y = _pyautogui.position()
                self.move(x - self.radius, y - self.radius)
                self.draw_circle()

            def draw_circle(self):
                img = _np.zeros((self.diameter, self.diameter, 4), dtype=_np.uint8)
                _cv2.circle(img, (self.radius + 2, self.radius + 2), max(1, self.radius - 5), (0, 255, 0, 255), 10)
                qimg = _QtGui.QImage(img.data, self.diameter, self.diameter, _QtGui.QImage.Format_RGBA8888)
                pixmap = _QtGui.QPixmap.fromImage(qimg)
                self.label.setPixmap(pixmap)

        app = _QtWidgets.QApplication(_sys.argv)
        overlay = CursorOverlay(radius=radius)
        overlay.show()
        _sys.exit(app.exec_())

    except Exception:
        # log exception trace for parent to inspect
        try:
            import traceback
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Cursor overlay crashed with exception:\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        raise

# ---------- Eye-control script (main process) ----------
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

# Config & globals
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X, CENTER_Y = MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2

mouse_control_enabled = True
filter_length = 8
EPS = 1e-8

# Blink tuning
BLINK_RATIO_THRESHOLD = 0.22
BLINK_DEBOUNCE_SEC = 0.45
CLOSED_FRAMES_THRESHOLD = 3
OPEN_FRAMES_THRESHOLD = 2

# Announcer
tts_engine = pyttsx3.init()
ANNOUNCE_DEBOUNCE = 1.2

def speak(text):
    def _s(t):
        try:
            tts_engine.say(t)
            tts_engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_s, args=(text,), daemon=True).start()

REGIONS = {
    "Start Button": (0, MONITOR_HEIGHT - 200, 200, MONITOR_HEIGHT),
    "Menu": (MONITOR_WIDTH - 300, 0, MONITOR_WIDTH, 200),
}
def get_region_label(x, y):
    for label, (x1, y1, x2, y2) in REGIONS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return label
    return None

def get_window_title_under_cursor():
    if not IS_WINDOWS:
        return None
    pt = wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
        return None
    hwnd = ctypes.windll.user32.WindowFromPoint(pt)
    if not hwnd:
        return None
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return None
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value

# MediaPipe & camera
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.45,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)

EYE_LANDMARKS = {
    "left": {"up": 159, "down": 145, "left": 33,  "right": 133, "center": 130},
    "right":{"up": 386, "down": 374, "left": 362, "right": 263, "center": 359}
}

# blink state
left_closed_frames = right_closed_frames = 0
left_open_frames = right_open_frames = 0
left_ready = right_ready = True
last_left_click_time = last_right_click_time = 0.0
swapped_mapping = False

# announcement globals (module-level)
last_announced_label = None
last_announce_time = 0.0

# mouse mover & smoothing globals
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()
running = True

# smoothing params (tweak to taste)
pyautogui.PAUSE = 0
SMOOTH_ALPHA = 0.18   # EMA alpha (0..1) - increase for snappier, decrease for smoother
MOVE_THRESHOLD = 3    # px threshold before moving real mouse

pos_now = pyautogui.position()
smoothed_mouse_target = [int(pos_now.x if hasattr(pos_now, 'x') else pos_now[0]),
                         int(pos_now.y if hasattr(pos_now, 'y') else pos_now[1])]

def mouse_mover():
    """
    Exponentially smooth the raw mouse_target and only move the system mouse
    when the filtered target moves beyond MOVE_THRESHOLD.
    """
    global smoothed_mouse_target
    while running:
        with mouse_lock:
            tx, ty = mouse_target[0], mouse_target[1]

        # EMA smoothing
        sx = int((1.0 - SMOOTH_ALPHA) * smoothed_mouse_target[0] + SMOOTH_ALPHA * tx)
        sy = int((1.0 - SMOOTH_ALPHA) * smoothed_mouse_target[1] + SMOOTH_ALPHA * ty)

        dx = sx - smoothed_mouse_target[0]
        dy = sy - smoothed_mouse_target[1]

        smoothed_mouse_target[0] = sx
        smoothed_mouse_target[1] = sy

        if math.hypot(dx, dy) > MOVE_THRESHOLD:
            try:
                pyautogui.moveTo(sx, sy)
            except Exception:
                pass

        time.sleep(0.012)

def stop_app():
    global running
    running = False
    print("Global exit hotkey pressed. Shutting down...")

# register global hotkey
try:
    keyboard.add_hotkey('ctrl+shift+q', stop_app)
    print("Registered global hotkey: Ctrl+Shift+Q to quit.")
except Exception as e:
    print("Could not register global hotkey. Use 'q' in window to quit.", e)

# start mouse mover thread
threading.Thread(target=mouse_mover, daemon=True).start()

def landmark_to_np(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w])

def safe_normalize(v):
    n = np.linalg.norm(v)
    if n < EPS:
        return v
    return v / n

def eye_blink_ratio(landmarks, up_idx, down_idx, left_idx, right_idx, w, h):
    up = landmark_to_np(landmarks[up_idx], w, h)
    down = landmark_to_np(landmarks[down_idx], w, h)
    left = landmark_to_np(landmarks[left_idx], w, h)
    right = landmark_to_np(landmarks[right_idx], w, h)
    vert = np.linalg.norm(up - down)
    hor = np.linalg.norm(left - right) + EPS
    return vert / hor

def calibrate_eye(target='left', duration=2.0):
    global swapped_mapping
    print(f"Calibration: you will blink your {target} eye when prompted. Starting in 1s...")
    speak(f"Prepare to blink your {target} eye in a moment.")
    time.sleep(1.0)
    print("Calibrating now — blink the requested eye once or twice during the next 2 seconds.")
    speak("Now blink.")
    end = time.time() + duration
    samples = []
    while time.time() < end and running:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark
            try:
                l = EYE_LANDMARKS["left"]
                r = EYE_LANDMARKS["right"]
                left_ratio = eye_blink_ratio(landmarks, l["up"], l["down"], l["left"], l["right"], w, h)
                right_ratio = eye_blink_ratio(landmarks, r["up"], r["down"], r["left"], r["right"], w, h)
                samples.append((left_ratio, right_ratio))
            except Exception:
                pass
        time.sleep(0.03)
    if not samples:
        print("Calibration failed: no face/landmarks captured. Try again.")
        speak("Calibration failed. Try again.")
        return
    left_avg = np.mean([s[0] for s in samples])
    right_avg = np.mean([s[1] for s in samples])
    print(f"Calibration averages — left_ratio: {left_avg:.3f}, right_ratio: {right_avg:.3f}")
    detected_landmark = 'left' if left_avg < right_avg else 'right'
    if target == 'left':
        swapped_mapping = (detected_landmark != 'left')
    else:
        swapped_mapping = (detected_landmark == 'left')
    print(f"Calibration result: detected landmark = {detected_landmark}. swapped_mapping = {swapped_mapping}")
    speak(f"Calibration complete. Mapping {'swapped' if swapped_mapping else 'normal'}.")

print("Controls: L = calibrate left-eye, R = calibrate right-eye, M = toggle manual mapping, F7 = toggle mouse, q = quit window, Ctrl+Shift+Q = global quit.")

# Main eye control loop
def eye_control_loop():
    global running, left_closed_frames, right_closed_frames
    global left_open_frames, right_open_frames, left_ready, right_ready
    global last_left_click_time, last_right_click_time, swapped_mapping
    global mouse_control_enabled
    global last_announced_label, last_announce_time

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w, _ = frame.shape
        debug_overlay = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # draw landmarks
            for i, lm in enumerate(landmarks):
                pt = landmark_to_np(lm, w, h)
                xpt, ypt = int(pt[0]), int(pt[1])
                if 0 <= xpt < w and 0 <= ypt < h:
                    cv2.circle(debug_overlay, (xpt, ypt), 2, (200,200,200), -1)

            # ratios
            try:
                l = EYE_LANDMARKS["left"]
                r = EYE_LANDMARKS["right"]
                left_ratio = eye_blink_ratio(landmarks, l["up"], l["down"], l["left"], l["right"], w, h)
                right_ratio = eye_blink_ratio(landmarks, r["up"], r["down"], r["left"], r["right"], w, h)
                left_center_x = landmark_to_np(landmarks[l["center"]], w, h)[0]
                right_center_x = landmark_to_np(landmarks[r["center"]], w, h)[0]
            except Exception:
                left_ratio = right_ratio = 1.0
                left_center_x = 0
                right_center_x = w

            now = time.time()

            # update counters
            if left_ratio < BLINK_RATIO_THRESHOLD:
                left_closed_frames += 1
                left_open_frames = 0
            else:
                left_open_frames += 1
                left_closed_frames = 0

            if right_ratio < BLINK_RATIO_THRESHOLD:
                right_closed_frames += 1
                right_open_frames = 0
            else:
                right_open_frames += 1
                right_closed_frames = 0

            both_closed = (left_closed_frames >= CLOSED_FRAMES_THRESHOLD) and (right_closed_frames >= CLOSED_FRAMES_THRESHOLD)

            # left-landmark blink event
            if left_closed_frames >= CLOSED_FRAMES_THRESHOLD:
                if left_ready and not both_closed:
                    if swapped_mapping:
                        if (now - last_right_click_time) > BLINK_DEBOUNCE_SEC:
                            try:
                                pyautogui.click(button='right')
                            except Exception:
                                pass
                            last_right_click_time = now
                    else:
                        if (now - last_left_click_time) > BLINK_DEBOUNCE_SEC:
                            try:
                                pyautogui.click(button='left')
                            except Exception:
                                pass
                            last_left_click_time = now
                    left_ready = False
            if left_open_frames >= OPEN_FRAMES_THRESHOLD:
                left_ready = True

            # right-landmark blink event
            if right_closed_frames >= CLOSED_FRAMES_THRESHOLD:
                if right_ready and not both_closed:
                    if swapped_mapping:
                        if (now - last_left_click_time) > BLINK_DEBOUNCE_SEC:
                            try:
                                pyautogui.click(button='left')
                            except Exception:
                                pass
                            last_left_click_time = now
                    else:
                        if (now - last_right_click_time) > BLINK_DEBOUNCE_SEC:
                            try:
                                pyautogui.click(button='right')
                            except Exception:
                                pass
                            last_right_click_time = now
                    right_ready = False
            if right_open_frames >= OPEN_FRAMES_THRESHOLD:
                right_ready = True

            if both_closed:
                left_ready = False
                right_ready = False

            # head/pointer mapping
            try:
                left = landmark_to_np(landmarks[234], w, h)
                right = landmark_to_np(landmarks[454], w, h)
                top = landmark_to_np(landmarks[10], w, h)
                bottom = landmark_to_np(landmarks[152], w, h)
                front = landmark_to_np(landmarks[1], w, h)
            except Exception:
                left = right = top = bottom = front = np.array([w/2, h/2, 0.0])

            right_axis = safe_normalize(right - left)
            up_axis = safe_normalize(top - bottom)
            forward_axis = safe_normalize(np.cross(right_axis, up_axis))
            forward_axis = -forward_axis
            center = (left + right + top + bottom + front) / 5.0
            half_depth = 80

            ray_origins.append(center)
            ray_directions.append(forward_axis)
            if len(ray_origins) > 0 and len(ray_directions) > 0:
                avg_origin = np.mean(np.array(ray_origins), axis=0)
                avg_dir = safe_normalize(np.mean(np.array(ray_directions), axis=0))
            else:
                avg_origin = center
                avg_dir = forward_axis

            reference_forward = np.array([0.0, 0.0, -1.0])
            xz = np.array([avg_dir[0], 0.0, avg_dir[2]])
            if np.linalg.norm(xz) < EPS:
                xz = np.array([0.0, 0.0, -1.0])
            else:
                xz /= np.linalg.norm(xz)
            yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz), -1.0, 1.0))
            if avg_dir[0] < 0:
                yaw_rad = -yaw_rad

            yz = np.array([0.0, avg_dir[1], avg_dir[2]])
            if np.linalg.norm(yz) < EPS:
                yz = np.array([0.0, 0.0, -1.0])
            else:
                yz /= np.linalg.norm(yz)
            pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz), -1.0, 1.0))
            if avg_dir[1] > 0:
                pitch_rad = -pitch_rad

            yaw_deg = math.degrees(yaw_rad)
            pitch_deg = math.degrees(pitch_rad)
            if yaw_deg < 0:
                yaw_deg = abs(yaw_deg)
            elif yaw_deg < 180:
                yaw_deg = 360 - yaw_deg
            if pitch_deg < 0:
                pitch_deg = 360 + pitch_deg

            yawDegrees = 20
            pitchDegrees = 10
            screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * MONITOR_WIDTH)
            screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)
            margin = 10
            screen_x = max(margin, min(MONITOR_WIDTH - margin, screen_x))
            screen_y = max(margin, min(MONITOR_HEIGHT - margin, screen_y))

            # Only update raw mouse_target if frame looks reliable
            valid_frame = True
            if not (0 <= screen_x <= MONITOR_WIDTH and 0 <= screen_y <= MONITOR_HEIGHT):
                valid_frame = False
            if not np.isfinite(screen_x) or not np.isfinite(screen_y):
                valid_frame = False

            if mouse_control_enabled and valid_frame:
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y

            # announce region / window
            label = get_region_label(screen_x, screen_y)
            if not label and IS_WINDOWS:
                win_title = get_window_title_under_cursor()
                if win_title:
                    label = f"Window: {win_title}"
            tnow = time.time()
            if label and (label != last_announced_label or (tnow - last_announce_time) > ANNOUNCE_DEBOUNCE):
                last_announced_label = label
                last_announce_time = tnow
                speak(label)
                print("[Announce]", label)

            # draw marker of cursor location on camera view
            sx = int(screen_x * w / MONITOR_WIDTH)
            sy = int(screen_y * h / MONITOR_HEIGHT)
            cv2.drawMarker(debug_overlay, (sx, sy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

            # debug text
            cv2.rectangle(debug_overlay, (5,5), (560,130), (0,0,0), -1)
            cv2.putText(debug_overlay, f"LeftRatio: {left_ratio:.3f} RightRatio: {right_ratio:.3f}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(debug_overlay, f"Closed L:{left_closed_frames} R:{right_closed_frames}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(debug_overlay, f"Mapping swapped: {swapped_mapping} (M to toggle)", (10,75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(debug_overlay, "Press L (left cal) R (right cal) M (toggle)", (10,105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        else:
            cv2.putText(debug_overlay, "No face detected - check camera index/lighting", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Eye-Control (Ctrl+Shift+Q to quit)", debug_overlay)

        # keyboard handling (non-blocking)
        if keyboard.is_pressed('f7'):
            mouse_control_enabled = not mouse_control_enabled
            print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
            time.sleep(0.25)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User pressed 'q' -> exiting.")
            break
        elif key == ord('m'):
            swapped_mapping = not swapped_mapping
            print("Manual toggle mapping. swapped_mapping =", swapped_mapping)
            speak(f"Mapping {'swapped' if swapped_mapping else 'normal'}")
        elif key == ord('l'):
            calibrate_eye('left', duration=2.0)
        elif key == ord('r'):
            calibrate_eye('right', duration=2.0)
        elif key == ord('c'):
            print("Calibration placeholder (alternate).")

    print("Eye loop exiting...")

# Entrypoint
def main():
    cursor_proc = None
    try:
        if sys.platform == "win32":
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    try:
        cursor_proc = multiprocessing.Process(target=run_cursor_overlay, args=(80,))
        cursor_proc.start()
        print(f"Started cursor overlay process (pid={cursor_proc.pid})")
    except Exception as e:
        print("Failed to start cursor overlay process:", e)

    try:
        eye_control_loop()
    finally:
        try:
            if cursor_proc is not None and cursor_proc.is_alive():
                cursor_proc.terminate()
                cursor_proc.join(timeout=2.0)
                print("Cursor overlay process terminated.")
        except Exception as e:
            print("Error terminating cursor process:", e)

    # cleanup
    global running
    running = False
    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()
    try:
        face_mesh.close()
    except:
        pass
    print("Exited cleanly.")

if __name__ == "__main__":
    main()
