# AAIMS — Automated Academic Integrity Monitoring System
## Comprehensive Study Guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [File Structure](#2-file-structure)
3. [Application Flow](#3-application-flow)
4. [Line-by-Line: `main.py`](#4-line-by-line-mainpy)
5. [Line-by-Line: `frontend/styles.py`](#5-line-by-line-frontendstylingpy)
6. [Line-by-Line: `frontend/notice_screen.py`](#6-line-by-line-frontendnotice_screenpy)
7. [Line-by-Line: `frontend/session_setup.py`](#7-line-by-line-frontendsession_setuppy)
8. [Line-by-Line: `frontend/calibration_screen.py`](#8-line-by-line-frontendcalibration_screenpy)
9. [Line-by-Line: `modules/face_absence.py`](#9-line-by-line-modulesface_absencepy)
10. [Line-by-Line: `modules/head_pose.py`](#10-line-by-line-moduleshead_posepy)
11. [Line-by-Line: `modules/mouth_detection.py`](#11-line-by-line-modulesmouth_detectionpy)
12. [Line-by-Line: `modules/aggregator.py`](#12-line-by-line-modulesaggregatorpy)
13. [Line-by-Line: `frontend/proctor_dashboard.py`](#13-line-by-line-frontendproctor_dashboardpy)
14. [Line-by-Line: `frontend/integrity_log_screen.py`](#14-line-by-line-frontendintegrity_log_screenpy)
15. [Key Algorithms Explained](#15-key-algorithms-explained)
16. [Event Logging System](#16-event-logging-system)
17. [Design Decisions and Trade-offs](#17-design-decisions-and-trade-offs)

---

## 1. Project Overview

AAIMS is a real-time, webcam-based academic integrity monitoring system. It runs locally on a proctoring workstation during a computer-based examination and automatically detects three categories of suspicious behavior:

| Behavior | What is Detected |
|---|---|
| **Head Turn** | Student looking away from the screen (left, right, or down) for more than ~0.67 seconds |
| **Face Absence** | Student's face not visible in the webcam frame for more than ~1 second |
| **Mouth Movement** | Repeated open-close mouth cycles consistent with talking (not a single yawn) |

All processing is done locally — no video is stored or transmitted. Only timestamped behavioral event logs are written to a CSV file.

### What AAIMS is NOT
- It does not record video.
- It does not take screenshots.
- It does not detect eye gaze or screen activity.
- It does not make automated academic decisions — flagged events are reviewed by the instructor.

---

## 2. File Structure

```
AAIMS/
│
├── main.py                          ← Entry point, launches screen chain
│
├── modules/
│   ├── face_absence.py              ← Haar Cascade face presence detector
│   ├── head_pose.py                 ← solvePnP head angle estimator
│   ├── mouth_detection.py           ← SVM-based mouth open/closed classifier
│   └── aggregator.py               ← Combines all three, manages logs
│
├── frontend/
│   ├── styles.py                    ← Shared color, font, and button constants
│   ├── notice_screen.py             ← Pre-exam consent/notice display
│   ├── session_setup.py             ← Proctor enters student/exam details
│   ├── calibration_screen.py        ← 100-frame neutral baseline capture
│   ├── proctor_dashboard.py         ← Live monitoring view
│   └── integrity_log_screen.py      ← Post-exam event log review
│
├── models/
│   ├── svm_mouth_classifier.pkl     ← Trained SVM (open vs closed)
│   └── svm_mouth_scaler.pkl         ← StandardScaler for MAR values
│
├── logs/
│   └── session_YYYY-MM-DD.csv       ← Auto-generated event log per day
```

---

## 3. Application Flow

Files execute in this chain. Each screen calls the next via a callback.

```
main.py
  │  creates: AaimsProcessor (loads all models once)
  │
  └─► NoticeScreen(on_acknowledge=launch_session_setup)
            │ student clicks "I Acknowledge"
            ▼
      SessionSetupScreen(on_start=launch_calibration)
            │ proctor fills form, clicks Start → produces config dict
            ▼
      CalibrationScreen(processor, config, on_done=launch_dashboard)
            │ 100 frames collected → sets pitch_baseline, yaw_baseline, mar_baseline
            ▼
      ProctorDashboard(processor, config)
            │  background thread → process_frame() every camera frame
            │     └─► face_absence.py + head_pose.py + mouth_detection.py
            │ proctor clicks End Session
            ▼
      IntegrityLogScreen(processor, config)
```

The `AaimsProcessor` is instantiated **once** in `main.py` and passed through every screen. Models are loaded once at startup; the same processor, log file, and detection state persist for the whole exam session.

---

## 4. Line-by-Line: `main.py`

**Purpose:** Entry point. Creates the processor, defines the screen chain as nested callbacks, and starts with the notice screen.

```python
import warnings                                           # standard library: suppress noisy terminal output
warnings.filterwarnings(                                  # MediaPipe prints protobuf deprecation warnings
    'ignore', category=UserWarning, module='google.protobuf')  # hide them so the terminal stays clean

from modules.aggregator import AaimsProcessor             # the central detection coordinator
from frontend.notice_screen import NoticeScreen           # screen 1 in the chain
from frontend.session_setup import SessionSetupScreen     # screen 2 in the chain
from frontend.calibration_screen import CalibrationScreen # screen 3 in the chain
from frontend.proctor_dashboard import ProctorDashboard   # screen 4 in the chain

processor = AaimsProcessor()
# instantiated once at module load time, before any GUI appears
# this triggers: HeadPoseDetector(), FaceAbsenceDetector(), MouthMovementDetector(),
#                MediaPipe FaceMesh init, CLAHE init, log file creation


def launch_notice():
    print('[AAIMS] Starting — notice screen.')
    app = NoticeScreen(on_acknowledge=launch_session_setup)
    # pass launch_session_setup as a function reference (callback)
    # NoticeScreen stores it and calls it when the student clicks Acknowledge
    app.mainloop()          # blocks here — runs Tkinter event loop until window closes


def launch_session_setup():
    # called automatically by NoticeScreen._acknowledge()
    print('[AAIMS] Notice acknowledged — session setup screen.')
    app = SessionSetupScreen(on_start=launch_calibration)
    app.mainloop()


def launch_calibration(config):
    # called by SessionSetupScreen._validate_and_start() with the config dict
    # config = {'student_id': ..., 'exam_name': ..., 'workstation': ..., ...}
    print(f'[AAIMS] Session configured — launching calibration.')
    app = CalibrationScreen(processor=processor, config=config, on_done=launch_dashboard)
    app.mainloop()


def launch_dashboard(config):
    # called by CalibrationScreen._finish() after 100 frames collected
    # at this point processor.head_pose.pitch_baseline, .yaw_baseline, and
    # processor.mouth.mar_baseline have all been set from calibration data
    print(f'[AAIMS] Calibration done — launching dashboard.')
    app = ProctorDashboard(processor=processor, config=config)
    app.mainloop()


if __name__ == '__main__':
    launch_notice()
    # only runs when executed directly (python main.py)
    # does NOT run if main.py is imported as a module
```

---

## 5. Line-by-Line: `frontend/styles.py`

**Purpose:** Centralizes all visual constants. Every screen imports from here so changing a color or font propagates everywhere.

```python
# ── Background colors (hex strings, used for bg= in tk widgets) ──────────────────
BG_MAIN    = '#F5F5F5'   # light gray — most screen backgrounds
BG_CARD    = '#EFEFEF'   # slightly darker — indicator rows, notice text panel
BG_HEADER  = '#2C3E50'   # dark navy — top header bar on the dashboard
BG_FLAGGED = '#FFE0E0'   # soft pink — flagged event rows in the integrity log table

# ── Text colors ───────────────────────────────────────────────────────────────────
TEXT_PRIMARY   = '#1A1A1A'   # near-black — main labels and values
TEXT_SECONDARY = '#444444'   # medium gray — subtitles, section labels

# ── Fonts (tuples: family, size, style) — passed as font= to tk.Label/Button ─────
FONT_TITLE      = ('Arial', 12, 'bold')   # screen title text e.g. "Session Setup"
FONT_HEADING    = ('Arial', 11, 'bold')   # notice screen main title (longer text)
FONT_SUBHEADING = ('Arial', 10, 'bold')   # section headings, primary action buttons
FONT_BODY       = ('Arial', 10)           # form labels and entry fields
FONT_LABEL      = ('Arial',  9, 'bold')   # card headers, "End Session" button
FONT_SMALL      = ('Arial',  9)           # status text, timestamps, descriptions
FONT_DOT        = ('Arial', 14)           # the ● status indicator — larger for visibility
FONT_MONO       = ('Courier', 9)          # metric values (yaw, pitch, MAR) — fixed-width

# ── Button style dicts — unpacked with ** at call sites ──────────────────────────
# Usage: tk.Button(parent, text='Label', **BTN_PRIMARY, pady=6, command=fn)
BTN_PRIMARY   = dict(bg='#2C3E50', fg='white', activebackground='#1A252F',
                     relief='flat', cursor='hand2')
# dark navy, white text, darker on hover/click, no border raised, pointer cursor

BTN_DANGER    = dict(bg='#7B0000', fg='white', activebackground='#5A0000',
                     relief='flat', cursor='hand2')
# dark red — used only for the "End Session" button

BTN_SECONDARY = dict(bg='#888888', fg='white', activebackground='#666666',
                     relief='flat', cursor='hand2')
# gray — used for "Close" button on integrity log screen
```

---

## 6. Line-by-Line: `frontend/notice_screen.py`

**Purpose:** Displays the pre-exam consent notice. No detection logic — purely informational UI.

```python
import tkinter as tk           # standard GUI toolkit — core widgets
from tkinter import ttk        # themed widget submodule: ttk.Separator (horizontal rule)
from frontend.styles import (  # import only what this screen uses
    BG_MAIN, BG_CARD,
    TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_HEADING, FONT_SUBHEADING, FONT_SMALL,
    BTN_PRIMARY
)

class NoticeScreen(tk.Tk):
    # subclasses tk.Tk (the root window)
    # each screen in this app is its own root window — they run sequentially, not simultaneously

    def __init__(self, on_acknowledge):
        super().__init__()                     # initializes the Tkinter root window
        self.on_acknowledge = on_acknowledge   # store callback — called when student clicks Acknowledge

        self.title('AAIMS — Pre-Examination Notice')   # text shown in the OS window title bar
        self.geometry('550x460')               # width=550, height=460 pixels
        self.resizable(False, False)           # lock both width and height
        self.configure(bg=BG_MAIN)             # set the window's background color

        tk.Label(
            self,                          # parent widget = the window itself
            text='Automated Academic Integrity Monitoring System',
            font=FONT_HEADING,
            bg=BG_MAIN, fg=TEXT_PRIMARY,
            wraplength=500                 # text wraps at 500px; prevents overflow on narrow windows
        ).pack(pady=(20, 2))               # pady=(top, bottom): 20px above, 2px below

        tk.Label(
            self, text='Pre-Examination Notice',
            font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_SECONDARY
        ).pack(pady=(0, 10))

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30)
        # horizontal divider line; fill='x' stretches it; padx=30 gives 30px margins each side

        notice_frame = tk.Frame(self, bg=BG_CARD)
        notice_frame.pack(fill='both', expand=True, padx=30, pady=10)
        # fill='both': frame grows in x and y; expand=True: takes all remaining vertical space

        notice = (
            'This workstation is currently being monitored by AAIMS.\n\n'
            '...'   # full notice text defined as a multiline string
        )

        tk.Label(
            notice_frame, text=notice,
            font=FONT_SMALL, bg=BG_CARD, fg=TEXT_PRIMARY,
            justify='left',               # left-align each line of text
            wraplength=460                # wrap text at 460px within the card frame
        ).pack(padx=15, pady=12)

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30)

        tk.Button(
            self, text='I Acknowledge — Proceed',
            font=FONT_SUBHEADING,
            **BTN_PRIMARY,                # unpacks: bg, fg, activebackground, relief, cursor
            pady=8, padx=20,
            command=self._acknowledge     # binds click to this method
        ).pack(pady=12)

    def _acknowledge(self):
        self.destroy()           # closes and frees this Tkinter window
        self.on_acknowledge()    # calls launch_session_setup() — opens next screen
```

---

## 7. Line-by-Line: `frontend/session_setup.py`

**Purpose:** Collects student ID, exam name, and workstation number from the proctor, then packages them into a `config` dict.

```python
import tkinter as tk
from tkinter import ttk, messagebox    # messagebox: pop-up dialog for validation error
from frontend.styles import (
    BG_MAIN, TEXT_PRIMARY,
    FONT_TITLE, FONT_SUBHEADING, FONT_BODY,
    BTN_PRIMARY
)

class SessionSetupScreen(tk.Tk):

    def __init__(self, on_start):
        super().__init__()
        self.on_start = on_start    # callback: called with config dict when proctor clicks Start

        self.title('AAIMS — Session Setup')
        self.geometry('480x340')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        tk.Label(self, text='Session Setup', font=FONT_TITLE, bg=BG_MAIN, fg=TEXT_PRIMARY).pack(pady=(20, 5))

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30, pady=(0, 15))

        form = tk.Frame(self, bg=BG_MAIN)
        form.pack(padx=40, fill='x')
        # frame holds the form in a grid layout (labels left, entries right)

        tk.Label(form, text='Student ID / Name:', font=FONT_BODY, bg=BG_MAIN, anchor='w').grid(
            row=0, column=0, sticky='w', pady=8)    # sticky='w': left-align within the grid cell
        self.student_id = tk.Entry(form, font=FONT_BODY, width=28, relief='solid', bd=1)
        self.student_id.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=8)
        # relief='solid', bd=1: draws a thin visible border around the entry field

        tk.Label(form, text='Exam / Session Name:', font=FONT_BODY, bg=BG_MAIN, anchor='w').grid(
            row=1, column=0, sticky='w', pady=8)
        self.exam_name = tk.Entry(form, font=FONT_BODY, width=28, relief='solid', bd=1)
        self.exam_name.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=8)
        self.exam_name.insert(0, 'Midterm Examination')    # pre-fill with default value

        tk.Label(form, text='Workstation Number:', font=FONT_BODY, bg=BG_MAIN, anchor='w').grid(
            row=2, column=0, sticky='w', pady=8)
        self.workstation = tk.Entry(form, font=FONT_BODY, width=28, relief='solid', bd=1)
        self.workstation.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=8)
        self.workstation.insert(0, '1')                    # pre-fill with default value

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30, pady=15)

        tk.Button(
            self, text='Start Baseline Calibration →',
            font=FONT_SUBHEADING, **BTN_PRIMARY, pady=6, padx=20,
            command=self._validate_and_start
        ).pack()

    def _validate_and_start(self):
        if not self.student_id.get().strip():
            # .get() reads the entry text; .strip() removes leading/trailing whitespace
            messagebox.showerror('Missing Information', 'Please enter the Student ID or Name.')
            return    # abort — do not proceed without a student ID

        config = {
            'student_id':        self.student_id.get().strip(),
            'exam_name':         self.exam_name.get().strip(),
            'workstation':       self.workstation.get().strip(),
            'yaw_threshold':     10,    # stored for metadata/display only — NOT fed into HeadPoseDetector
            'absence_threshold': 30     # stored for metadata/display only — NOT fed into FaceAbsenceDetector
        }
        # the thresholds are hardcoded inside the detector __init__ methods
        # they appear here only as informational fields in the session record

        print(f'[Session] Student: {config["student_id"]} | Exam: {config["exam_name"]} | Workstation: {config["workstation"]}')

        self.destroy()         # close this window
        self.on_start(config)  # pass config dict forward to launch_calibration()
```

---

## 8. Line-by-Line: `frontend/calibration_screen.py`

**Purpose:** Captures 100 webcam frames while the student sits neutrally. Computes mean pitch, yaw, and MAR baselines and writes them into the processor.

```python
import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from frontend.styles import (BG_MAIN, TEXT_PRIMARY, TEXT_SECONDARY,
                              FONT_TITLE, FONT_SUBHEADING, FONT_SMALL, BTN_PRIMARY)
from PIL import Image, ImageTk    # PIL: convert numpy frame to Tkinter-compatible image

CALIBRATION_FRAMES = 100    # module-level constant: how many frames to collect for baseline
# placed outside the class so it can be changed without touching __init__

class CalibrationScreen(tk.Tk):

    def __init__(self, processor, config, on_done):
        super().__init__()
        self.processor = processor   # AaimsProcessor — baselines will be written into this
        self.config    = config      # session metadata dict (passed forward to dashboard)
        self.on_done   = on_done     # callback: called with config when calibration finishes

        self.title('AAIMS — Baseline Calibration')
        self.geometry('540x480')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        tk.Label(self, text='Baseline Calibration', font=FONT_TITLE, bg=BG_MAIN, fg=TEXT_PRIMARY).pack(pady=(12, 2))
        tk.Label(self, text='Sit naturally, face the camera, and keep your mouth closed.',
                 font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack()

        self.video_label = tk.Label(self, bg='black', width=480, height=300)
        self.video_label.pack(pady=8)    # webcam preview is drawn here

        self.progress_label = tk.Label(self, text='Press Start when you are ready.',
                                       font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_PRIMARY)
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self, length=400, maximum=CALIBRATION_FRAMES)
        self.progress_bar.pack(pady=6)    # fills left-to-right as frames are collected

        self.start_btn = tk.Button(self, text='Start Calibration', font=FONT_SUBHEADING,
                                   **BTN_PRIMARY, pady=5, padx=15, command=self._start)
        self.start_btn.pack(pady=6)

        self.cap = cv2.VideoCapture(0)    # open default webcam (must match dashboard's camera index)
        self.calibrating = False          # flag: True only after Start button is clicked
        self.angle_buffer = []            # list of (pitch, yaw) tuples collected so far
        self.mar_buffer   = []            # list of float MAR values collected so far
        self.captured     = 0             # count of frames successfully processed
        self.running      = True          # preview loop runs while True

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # a separate FaceMesh instance just for calibration
        # the main processor.face_mesh is reserved for the live session

        self.protocol('WM_DELETE_WINDOW', self._on_close)
        # intercept the X button: release camera before destroying window

        self._update_preview()    # start the preview loop immediately (before Start is clicked)


    def _update_preview(self):
        if not self.running:
            return    # stop loop if window is closing

        success, frame = self.cap.read()
        if success:
            rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (480, 300))
            display_image = ImageTk.PhotoImage(Image.fromarray(resized_frame))
            self.video_label.configure(image=display_image)
            self.video_label.image = display_image    # keep reference to prevent garbage collection

            if self.calibrating:
                self._process_calibration_frame(frame)    # collect data only after Start is clicked

        self.after(33, self._update_preview)
        # schedule next call 33ms later (~30fps)
        # self.after is non-blocking: Tkinter event loop continues between calls


    def _on_close(self):
        self.running = False       # stop the preview loop
        self.cap.release()         # release webcam hardware
        self.destroy()             # close window


    def _start(self):
        self.calibrating = True                                         # enable data collection
        self.start_btn.configure(state='disabled')                      # prevent double-click
        self.progress_label.configure(text='Calibrating — please stay still...')


    def _process_calibration_frame(self, frame):
        if self.captured >= CALIBRATION_FRAMES:
            return    # guard: do not process frames after buffer is full

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.face_mesh.process(rgb_frame)
        # runs MediaPipe FaceMesh on this frame

        if results.multi_face_landmarks:
            # only count this frame if MediaPipe found a face
            landmarks = results.multi_face_landmarks[0]

            angles = self.processor.head_pose.get_angles(landmarks, frame_width, frame_height)
            # returns (pitch, yaw, roll, nose_2d, nose_3d, rot_vec, trans_vec)
            if angles[0] is not None:
                # angles[0] is pitch; None means solvePnP failed this frame
                self.angle_buffer.append((angles[0], angles[1]))    # store (pitch, yaw)

            mouth_aspect_ratio = self.processor.mouth.compute_mar(
                landmarks.landmark, frame_width, frame_height)
            self.mar_buffer.append(mouth_aspect_ratio)    # store MAR for closed-mouth baseline

            self.captured += 1
            self.progress_bar['value'] = self.captured
            self.progress_label.configure(text=f'Calibrating... {self.captured}/{CALIBRATION_FRAMES} frames')

        if self.captured >= CALIBRATION_FRAMES:
            self._finish()    # all frames collected → compute baselines


    def _finish(self):
        self.processor.head_pose.calibrate(self.angle_buffer)
        # computes: pitch_baseline = mean of all collected pitches
        #           yaw_baseline   = mean of all collected yaws
        # these values shift the origin of head turn detection to the student's natural position

        if self.mar_buffer:
            self.processor.mouth.mar_baseline = float(np.mean(self.mar_buffer))
            # mean of 100 closed-mouth MAR values = the student's resting MAR
            # used as the gate threshold: MAR must exceed mar_baseline × 1.5 to count as "open"

        print(f'Calibration complete.')
        print(f'Head baseline pitch: {self.processor.head_pose.pitch_baseline:.2f}')
        print(f'Head baseline yaw:   {self.processor.head_pose.yaw_baseline:.2f}')
        print(f'MAR baseline:        {self.processor.mouth.mar_baseline:.4f}')

        self.progress_label.configure(text='Calibration complete!', fg='green')

        self.running = False
        self.cap.release()    # release webcam — dashboard will open its own capture

        self.after(1200, lambda: [
            self.destroy(),
            self.on_done(self.config)    # calls launch_dashboard(config) after 1.2 second delay
        ])
        # lambda groups two statements; delay gives the "Calibration complete!" text time to display
```

---

## 9. Line-by-Line: `modules/face_absence.py`

**Purpose:** Wraps OpenCV's Haar Cascade for face presence detection. Called by the aggregator once per frame to contribute to the dual-confirmation absence counter.

```python
import cv2    # OpenCV: provides CascadeClassifier and detectMultiScale

class FaceAbsenceDetector:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +                   # path to OpenCV's bundled XML files
            'haarcascade_frontalface_default.xml'     # the pre-trained frontal face model
        )
        # loads the cascade classifier from XML into memory
        # Haar Cascade is a classical ML method: scans image at multiple scales,
        # passes regions through a cascade of simple feature detectors

        self.absence_counter   = 0    # increments every frame both detectors miss the face
        self.ABSENCE_THRESHOLD = 30   # 30 consecutive unconfirmed frames ≈ 1 second → absence_flag
        print('FaceAbsenceDetector loaded successfully.')

    def predict(self, grayscale_frame):
        # grayscale_frame arrives preprocessed from aggregator:
        # resized to 640×480, Gaussian blurred, CLAHE enhanced
        faces = self.face_cascade.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.05,     # shrink image by 5% each scale level — more thorough than 1.1
            minNeighbors=4,       # detection rectangle needs 4 overlapping neighbors → reduces noise
            minSize=(30, 30)      # ignore regions smaller than 30×30 pixels
        )
        # faces: numpy array of shape (N, 4) where each row is [x, y, w, h] of a detected face
        #        OR empty tuple () if nothing detected
        return 1 if len(faces) > 0 else 0    # 1 = face found, 0 = no face

    def reset(self):
        self.absence_counter = 0    # utility method for manual reset during testing
```

**Key limitation:** Haar Cascade is trained on upright frontal faces. Significant head tilt causes it to miss faces. This is why the aggregator uses it only as one half of the AND gate — MediaPipe handles tilted faces.

---

## 10. Line-by-Line: `modules/head_pose.py`

**Purpose:** Takes 6 MediaPipe landmarks and solves for Euler angles using PnP. Compares angles against calibrated baselines to detect sustained off-screen head turns.

```python
import cv2
import numpy as np

class HeadPoseDetector:

    def __init__(self):
        self.yaw_threshold   = 10    # degrees of yaw deviation from baseline to start counting
        self.pitch_threshold = 10    # degrees of pitch deviation from baseline to start counting

        self.pitch_baseline = 0      # set during calibration; 0 = straight-ahead at center screen
        self.yaw_baseline   = 0      # set during calibration; 0 = straight-ahead at center screen

        self.sustained_off_screen_frames = 0   # rolling count of consecutive off-screen frames
        self.FRAMES_TO_FLAG_HEAD_TURN    = 20  # 20 frames ≈ 0.67s at 30fps → head_flag = True

    def get_angles(self, face_landmarks, image_width, image_height):
        face_2d = []     # 2D pixel coordinates: shape (6, 2)
        face_3d = []     # pseudo-3D coordinates: shape (6, 3) using MediaPipe's z
        nose_2d = None   # pixel coord of nose tip — stored for optional visualization
        nose_3d = None   # pseudo-3D coord of nose tip

        for landmark_index, landmark in enumerate(face_landmarks.landmark):
            # face_landmarks.landmark is a list of 478 landmark objects
            if landmark_index in [1, 33, 61, 199, 263, 291]:
                # 6 chosen landmarks: nose tip (1), eye outer corners (33, 263),
                # mouth corners (61, 291), chin (199)
                # chosen because they form a stable, spread configuration for solvePnP

                if landmark_index == 1:
                    # store nose tip separately (index 1 = nose tip in MediaPipe)
                    nose_2d = (landmark.x * image_width, landmark.y * image_height)
                    nose_3d = (landmark.x * image_width, landmark.y * image_height, landmark.z * 3000)
                    # z * 3000 scales the very small normalized depth to a usable range

                x_pixel = int(landmark.x * image_width)     # normalized x (0–1) → pixel x
                y_pixel = int(landmark.y * image_height)    # normalized y (0–1) → pixel y
                face_2d.append([x_pixel, y_pixel])
                face_3d.append([x_pixel, y_pixel, landmark.z])
                # landmark.z is MediaPipe's relative depth; used as pseudo-Z

        face_2d = np.array(face_2d, dtype=np.float64)    # solvePnP requires float64
        face_3d = np.array(face_3d, dtype=np.float64)    # solvePnP requires float64

        focal_length = image_width
        # simplified pinhole camera model: focal_length ≈ image width in pixels
        # this is an approximation used when real camera calibration data is unavailable

        camera_matrix = np.array([
            [focal_length, 0,            image_height / 2],
            # row 0: [fx,  0, cx] where fx = focal length, cx = principal point x
            [0,            focal_length, image_width  / 2],
            # row 1: [0, fy, cy] where fy = focal length, cy = principal point y
            [0,            0,            1                ]
            # row 2: homogeneous coordinate row — always [0, 0, 1] for pinhole model
        ])
        # note: cx uses image_height, cy uses image_width — matches the axis convention here

        distortion_coefficients = np.zeros((4, 1), dtype=np.float64)
        # assume no lens distortion; reasonable for typical webcams at arm's length

        solved, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d,                   # 3D points (object space)
            face_2d,                   # corresponding 2D projections (image space)
            camera_matrix,             # intrinsic camera parameters
            distortion_coefficients    # assumed zero distortion
        )
        # rotation_vector: shape (3, 1), axis-angle representation (Rodrigues form)
        # translation_vector: shape (3, 1), face position relative to camera

        if not solved:
            return None, None, None, None, None, None, None
            # signals failure to caller; aggregator checks pitch is not None before using

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # converts (3,1) Rodrigues axis-angle vector → (3,3) rotation matrix

        euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        # RQ decomposition: factors the rotation matrix into Euler angles
        # output values are very small (near 0.0) due to the input scale

        pitch = euler_angles[0] * 360    # x-axis rotation in degrees; positive = looking up
        yaw   = euler_angles[1] * 360    # y-axis rotation in degrees; positive = looking left
        roll  = euler_angles[2] * 360    # z-axis rotation; head tilt — computed but not flagged

        return pitch, yaw, roll, nose_2d, nose_3d, rotation_vector, translation_vector

    def get_direction(self, pitch, yaw):
        # uses raw angles against the fixed threshold (no baseline subtraction here)
        # direction label is for display only; flagging uses detect_head_turned()
        if yaw > self.yaw_threshold:           return "Looking Left"
        elif yaw < -self.yaw_threshold:        return "Looking Right"
        elif pitch < -self.pitch_threshold:    return "Looking Down"
        elif pitch > self.pitch_threshold:     return "Looking Up"
        else:                                  return "Forward"

    def calibrate(self, angle_buffer):
        if not angle_buffer:
            return    # guard: if no frames collected (face never detected), skip calibration
        self.pitch_baseline = np.mean([a[0] for a in angle_buffer])
        # list comprehension extracts pitch from each (pitch, yaw) tuple; np.mean averages them
        self.yaw_baseline   = np.mean([a[1] for a in angle_buffer])
        print(f'Head baseline — pitch: {self.pitch_baseline:.1f}  yaw: {self.yaw_baseline:.1f}')

    def detect_head_turned(self, pitch, yaw):
        pitch_deviation = pitch - self.pitch_baseline
        # positive = looking more upward than neutral; negative = looking more downward
        yaw_deviation = abs(yaw - self.yaw_baseline)
        # abs() because both left (positive yaw) and right (negative yaw) are suspicious

        horizontal_turn = yaw_deviation   > self.yaw_threshold     # turned left or right?
        looking_down    = pitch_deviation < -self.pitch_threshold   # head tilted down?
        # looking UP is intentionally NOT flagged — proctors stand above students

        if horizontal_turn or looking_down:
            self.sustained_off_screen_frames += 1    # still off-screen → keep counting
        else:
            self.sustained_off_screen_frames = 0     # back to neutral → reset immediately

        return self.sustained_off_screen_frames >= self.FRAMES_TO_FLAG_HEAD_TURN
        # True only after 20 consecutive off-screen frames (≈0.67 seconds)
```

---

## 11. Line-by-Line: `modules/mouth_detection.py`

**Purpose:** Computes Mouth Aspect Ratio, classifies it with a trained SVM, and counts open-close cycles to distinguish talking from yawning. Maintains a sustained `is_talking` state that persists until 30 frames of inactivity.

```python
import numpy as np
import joblib      # for deserializing sklearn model objects from .pkl files

# MediaPipe FaceMesh landmark indices for mouth geometry
UPPER_LIP    = 13     # upper lip center
LOWER_LIP    = 14     # lower lip center
LEFT_CORNER  = 61     # left mouth corner
RIGHT_CORNER = 291    # right mouth corner

class MouthMovementDetector:

    def __init__(
        self,
        model_path ='models/svm_mouth_classifier.pkl',
        scaler_path='models/svm_mouth_scaler.pkl'
    ):
        self.mouth_classifier = joblib.load(model_path)
        # deserializes the trained SVM object from disk
        # SVM was trained on MAR values: 0 = closed, 1 = open

        self.feature_scaler = joblib.load(scaler_path)
        # deserializes the fitted StandardScaler from disk
        # scaler stores the training mean and std; applied to new MAR values before SVM

        # ── Open-close cycle counting ─────────────────────────────────────────────────
        self.open_frame_count          = 0   # consecutive frames mouth has been open this cycle
        self.MIN_OPEN_FRAMES_PER_CYCLE = 5   # mouth must stay open for 5+ frames to count as a cycle

        self.open_close_cycle_count = 0   # number of valid open-close cycles accumulated
        self.CYCLES_TO_FLAG_TALKING = 3   # 3 cycles → set is_talking = True

        # ── State tracking ────────────────────────────────────────────────────────────
        self.last_mar             = 0.0   # most recent computed MAR; read by aggregator for GUI
        self.current_mouth_state  = 0     # SVM+gate output this frame: 1=open, 0=closed
        self.previous_mouth_state = 0     # output from previous frame; detects open/close transitions
        self.mar_baseline         = 0.0   # set by CalibrationScreen; resting (closed) MAR
        self.MAR_OPEN_MULTIPLIER  = 1.5   # MAR must exceed mar_baseline × 1.5 to accept SVM "open"

        # ── Sustained talking state ───────────────────────────────────────────────────
        self.is_talking                   = False  # True while a talking episode is active
        self.talking_inactivity_counter   = 0      # frames elapsed since last completed cycle
        self.TALKING_INACTIVITY_THRESHOLD = 30     # 30 consecutive inactive frames → is_talking = False

        print('MouthMovementDetector loaded successfully.')


    def compute_mar(self, landmarks, image_width, image_height):
        def landmark_to_pixel(landmark_index):
            landmark = landmarks[landmark_index]
            return np.array([landmark.x * image_width, landmark.y * image_height])
            # converts normalized (0-1) MediaPipe landmark to pixel coordinates

        vertical_lip_distance = np.linalg.norm(
            landmark_to_pixel(UPPER_LIP) - landmark_to_pixel(LOWER_LIP))
        # Euclidean distance between upper and lower lip centers (height of the mouth opening)

        horizontal_mouth_width = np.linalg.norm(
            landmark_to_pixel(LEFT_CORNER) - landmark_to_pixel(RIGHT_CORNER))
        # Euclidean distance between left and right corners (width of the mouth)

        return vertical_lip_distance / (horizontal_mouth_width + 1e-6)
        # MAR = opening height ÷ width
        # dividing by width normalizes for face size and camera distance
        # + 1e-6 prevents division-by-zero on degenerate frames where width rounds to zero


    def detect_talking(self, landmarks, image_width, image_height):
        mouth_aspect_ratio = self.compute_mar(landmarks, image_width, image_height)
        self.last_mar = mouth_aspect_ratio    # store for aggregator to include in GUI result dict

        # ── SVM classification ────────────────────────────────────────────────────────
        normalized_mar = self.feature_scaler.transform([[mouth_aspect_ratio]])
        # StandardScaler: (x - training_mean) / training_std
        # puts the MAR value on the same scale as the training data
        svm_prediction = self.mouth_classifier.predict(normalized_mar)[0]
        # → 0 (closed) or 1 (open)

        # ── Calibration baseline gate ─────────────────────────────────────────────────
        baseline_open = (
            self.mar_baseline > 0 and                                    # calibration was performed
            mouth_aspect_ratio > self.mar_baseline * self.MAR_OPEN_MULTIPLIER
            # current MAR must exceed 1.5× the student's resting MAR
        )
        self.current_mouth_state = 1 if (
            svm_prediction == 1 and (self.mar_baseline == 0.0 or baseline_open)
        ) else 0
        # dual-gate logic:
        #   if calibration was done: both SVM AND baseline gate must agree
        #   if calibration was skipped (mar_baseline == 0.0): SVM alone decides

        cycle_completed = False    # tracks if a valid open-close cycle finished this frame

        # ── Cycle counting ────────────────────────────────────────────────────────────
        if self.current_mouth_state == 1:    # mouth is open this frame
            self.open_frame_count += 1
            if self.previous_mouth_state == 0:
                # transition: closed → open; print only on the first open frame
                print(f'[Mouth] Opened  | MAR: {mouth_aspect_ratio:.4f} | cycle: {self.open_close_cycle_count}/{self.CYCLES_TO_FLAG_TALKING}')
        else:    # mouth is closed this frame
            if self.open_frame_count >= self.MIN_OPEN_FRAMES_PER_CYCLE:
                # mouth was open long enough to count as a valid cycle
                self.open_close_cycle_count += 1
                cycle_completed = True
                print(f'[Mouth] Cycle {self.open_close_cycle_count}/{self.CYCLES_TO_FLAG_TALKING} completed (open for {self.open_frame_count} frames)')

                if self.open_close_cycle_count >= self.CYCLES_TO_FLAG_TALKING:
                    # 3 complete cycles reached → start/sustain talking episode
                    self.is_talking = True
                    self.open_close_cycle_count = 0     # reset for detecting next episode
                    self.talking_inactivity_counter = 0
                    print(f'[Mouth] TALKING FLAGGED — cycles threshold reached')

            elif self.open_frame_count > 0:
                # mouth opened but closed too quickly → dismiss as noise (blink, breath)
                print(f'[Mouth] Dismissed — only open for {self.open_frame_count} frames (min: {self.MIN_OPEN_FRAMES_PER_CYCLE})')

            self.open_frame_count = 0    # always reset when mouth closes

        # ── Sustained state management ────────────────────────────────────────────────
        if self.is_talking:
            if cycle_completed:
                self.talking_inactivity_counter = 0   # new activity → reset decay timer
            else:
                self.talking_inactivity_counter += 1  # no new cycle this frame → count inactivity

            if self.talking_inactivity_counter >= self.TALKING_INACTIVITY_THRESHOLD:
                # 30 frames passed without a new cycle completing → talking episode ended
                self.is_talking = False
                self.talking_inactivity_counter = 0
                print(f'[Mouth] Talking episode ended — inactivity threshold reached')

        self.previous_mouth_state = self.current_mouth_state
        # save for next frame's transition detection (closed→open print trigger)

        return self.is_talking
        # True = currently in a talking episode (sustained until 30 inactive frames)
        # False = not talking


    def reset(self):
        self.open_frame_count           = 0
        self.open_close_cycle_count     = 0
        self.last_mar                   = 0.0
        self.current_mouth_state        = 0
        self.is_talking                 = False
        self.talking_inactivity_counter = 0
```

**Why `is_talking` is sustained (not one-shot):** If `detect_talking` returned `True` for only one frame (when the 3rd cycle completes), `_start_episode` and `_end_episode` would fire on back-to-back frames, always logging `duration=0` seconds. The sustained state keeps `mouth_flag` True for the entire talking episode so the CSV row records a meaningful duration.

---

## 12. Line-by-Line: `modules/aggregator.py`

**Purpose:** Central coordinator. Owns all three detectors, the MediaPipe FaceMesh instance, and the event log. `process_frame()` is called every frame by the dashboard thread.

```python
import cv2
import mediapipe as mp
import csv
import datetime
import os
import time

from modules.head_pose import HeadPoseDetector
from modules.face_absence import FaceAbsenceDetector
from modules.mouth_detection import MouthMovementDetector

EVENT_LABELS = {
    'HEAD_TURN':   'Student turned head away from screen',
    'FACE_ABSENT': 'Student face not visible in camera',
    'MOUTH_OPEN':  'Repeated mouth movement detected',
}
# maps event type keys (used in code) to human-readable descriptions (written to CSV)


class AaimsProcessor:

    def __init__(self):
        self.head_pose    = HeadPoseDetector()       # loads angle thresholds, zeroes counters
        self.face_absence = FaceAbsenceDetector()    # loads Haar Cascade XML into memory
        self.mouth        = MouthMovementDetector()  # loads SVM + StandardScaler .pkl files

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,                  # track only one student per workstation
            min_detection_confidence=0.5,     # minimum score to detect a new face
            min_tracking_confidence=0.5       # minimum score to continue tracking a known face
        )
        # loads a TFLite neural network into memory; processes RGB frames, returns 478 landmarks

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        # clipLimit=2.0: cap the contrast amplification to avoid over-enhancing noise
        # tileGridSize=(8,8): divide image into 8×8 non-overlapping tiles; equalize each locally
        # applied to grayscale frame so Haar Cascade works better in uneven lighting

        self.alert_callback = None
        # set externally by ProctorDashboard: self.processor.alert_callback = self._show_alert
        # called once per episode start to trigger the red alert banner

        self.active_episodes = {
            'HEAD_TURN':   None,    # None = not currently active
            'FACE_ABSENT': None,    # datetime = start time of an active episode
            'MOUTH_OPEN':  None,
        }

        self.mouth_alert_time  = 0   # timestamp of last mouth_flag; drives 5s display timer
        self.MOUTH_ALERT_DISPLAY = 5 # seconds to keep mouth indicator red after detection

        os.makedirs('logs', exist_ok=True)    # create logs/ directory if it doesn't exist yet
        self.log_path = f'logs/session_{datetime.date.today()}.csv'
        # one log file per calendar day e.g. logs/session_2026-05-26.csv
        self._init_log()    # write CSV header immediately

        print('AaimsProcessor initialized successfully.')


    def _init_log(self):
        with open(self.log_path, 'w', newline='') as f:
            # 'w' mode: creates file or overwrites existing file for today
            writer = csv.writer(f)
            writer.writerow(['start_time', 'end_time', 'duration_seconds', 'event_type', 'description'])
            # header row; the file exists with only this row until detection begins


    def _start_episode(self, event_type):
        if self.active_episodes[event_type] is None:
            # only start if this event type is not already active
            self.active_episodes[event_type] = datetime.datetime.now()
            # record start timestamp
            print(f'[Episode START] {event_type} at {self.active_episodes[event_type].strftime("%H:%M:%S")}')
            if self.alert_callback:
                self.alert_callback(EVENT_LABELS[event_type])
                # fires alert banner on the dashboard — exactly once per episode


    def _end_episode(self, event_type):
        start    = self.active_episodes[event_type]    # datetime stored by _start_episode
        end      = datetime.datetime.now()
        duration = (end - start).seconds               # integer seconds

        with open(self.log_path, 'a', newline='') as f:
            # 'a' mode: append to file — preserves all previously logged episodes
            writer = csv.writer(f)
            writer.writerow([
                start.strftime('%Y-%m-%d %H:%M:%S'),    # formatted start time
                end.strftime('%Y-%m-%d %H:%M:%S'),      # formatted end time
                duration,                               # integer seconds
                event_type,                             # 'HEAD_TURN', 'FACE_ABSENT', 'MOUTH_OPEN'
                EVENT_LABELS[event_type]                # human-readable description
            ])

        print(f'[Episode END]   {event_type} | duration: {duration}s | logged to {self.log_path}')
        self.active_episodes[event_type] = None    # mark episode as closed


    def _update_episodes(self, head_flag, absence_flag, mouth_flag):
        flags = {
            'HEAD_TURN':   head_flag,      # True if head turned for 20+ consecutive frames
            'FACE_ABSENT': absence_flag,   # True if face unconfirmed for 30+ consecutive frames
            'MOUTH_OPEN':  mouth_flag,     # True while is_talking is active (sustained)
        }

        for event_type, is_active in flags.items():
            if is_active:
                self._start_episode(event_type)
                # no-op if already active (active_episodes[type] is not None)
            else:
                if self.active_episodes[event_type] is not None:
                    # was active, now stopped → write CSV row
                    self._end_episode(event_type)


    def process_frame(self, frame):
        image_height, image_width = frame.shape[:2]
        # frame.shape = (height, width, channels); [:2] discards channels

        # ── Haar Cascade preprocessing ────────────────────────────────────────────────
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # remove color channels; Haar only needs intensity values
        grayscale_frame = cv2.resize(grayscale_frame, (640, 480))
        # standardize input size for Haar Cascade — model was trained on similar scale faces
        grayscale_frame = cv2.GaussianBlur(grayscale_frame, (5, 5), 0)
        # kernel (5,5) smooths fine noise; sigmaX=0 → auto-calculated from kernel size
        grayscale_frame = self.clahe.apply(grayscale_frame)
        # local contrast enhancement: makes faces in shadows or bright spots easier to detect

        # ── MediaPipe preprocessing ────────────────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # OpenCV reads BGR; MediaPipe expects RGB — must swap the R and B channels

        # ── Initialize default values (overwritten if face is detected) ────────────────
        pitch = 0.0;  yaw = 0.0;  roll = 0.0
        mouth_aspect_ratio = 0.0
        face_present = 0
        head_flag    = False
        mouth_flag   = False
        direction    = 'No Face'

        # ── Run both detectors ─────────────────────────────────────────────────────────
        results = self.face_mesh.process(rgb_frame)
        mediapipe_detects_face = bool(results.multi_face_landmarks)
        # multi_face_landmarks is None if no face, list with one entry if face found

        haar_detects_face = self.face_absence.predict(grayscale_frame) == 1
        # predict() returns int 1 or 0; == 1 converts to bool

        # ── Absence counter (AND logic) ────────────────────────────────────────────────
        if haar_detects_face and mediapipe_detects_face:
            self.face_absence.absence_counter = 0
            # both agree face is present → strong confirmation → reset counter
        else:
            self.face_absence.absence_counter += 1
            # either missed → uncertain → increment; after 30 frames triggers absence_flag

        # ── Per-frame detection (only when MediaPipe found landmarks) ─────────────────
        if mediapipe_detects_face:
            face_present = 1
            landmarks    = results.multi_face_landmarks[0]
            # index [0] = first (only) face since max_num_faces=1

            pitch, yaw, roll, *_ = self.head_pose.get_angles(landmarks, image_width, image_height)
            # *_ discards nose_2d, nose_3d, rotation_vector, translation_vector (not needed here)

            if pitch is not None:
                # None if solvePnP failed (degenerate landmark configuration)
                direction = self.head_pose.get_direction(pitch, yaw)
                head_flag = self.head_pose.detect_head_turned(pitch, yaw)

            mouth_flag         = self.mouth.detect_talking(landmarks.landmark, image_width, image_height)
            mouth_aspect_ratio = self.mouth.last_mar    # MAR computed inside detect_talking()

        else:
            # MediaPipe found no face → reset all mouth detection state
            face_present = 0
            direction    = 'No Face'

            self.mouth.last_mar                   = 0.0
            self.mouth.current_mouth_state        = 0
            self.mouth.open_frame_count           = 0
            self.mouth.open_close_cycle_count     = 0
            self.mouth.is_talking                 = False
            self.mouth.talking_inactivity_counter = 0
            mouth_aspect_ratio = 0.0
            # prevents stale open-mouth state from being carried into frames after face disappears

        # ── Absence flag evaluation ────────────────────────────────────────────────────
        absence_flag = (self.face_absence.absence_counter >= self.face_absence.ABSENCE_THRESHOLD)
        # True after 30 consecutive unconfirmed frames; stays True until counter resets

        # ── Episode tracking ───────────────────────────────────────────────────────────
        self._update_episodes(head_flag, absence_flag, mouth_flag)
        # opens/closes log episodes; writes CSV rows when flags turn off

        # ── Mouth alert display timer ──────────────────────────────────────────────────
        if mouth_flag:
            self.mouth_alert_time = time.time()
            # record when talking was last active
        mouth_alert_active = (time.time() - self.mouth_alert_time) < self.MOUTH_ALERT_DISPLAY
        # keep indicator red for 5 seconds after the last active mouth_flag frame

        return {
            'pitch':              pitch,               # float, degrees
            'yaw':                yaw,                 # float, degrees
            'roll':               roll,                # float, degrees (not used for flagging)
            'mar':                mouth_aspect_ratio,  # float, for dashboard display
            'face_present':       face_present,        # 0 or 1, immediate single-frame value
            'direction':          direction,           # "Forward", "Looking Left", etc.
            'head_flag':          head_flag,           # bool, True if turned 20+ consecutive frames
            'absence_flag':       absence_flag,        # bool, True if absent 30+ consecutive frames
            'mouth_flag':         mouth_flag,          # bool, True while is_talking is active
            'mouth_alert_active': mouth_alert_active,  # bool, True for 5s after last mouth event
            'flagged':            head_flag or absence_flag or mouth_flag
        }
```

---

## 13. Line-by-Line: `frontend/proctor_dashboard.py`

**Purpose:** Main monitoring screen. Runs a background thread for the camera loop. Receives results from `process_frame()` and updates the GUI. Handles session end and transitions to the log screen.

```python
import tkinter as tk
from tkinter import ttk
import cv2
import threading          # for running the camera loop without blocking the UI
from PIL import Image, ImageTk
from frontend.styles import (BG_MAIN, BG_CARD, BG_HEADER, TEXT_PRIMARY, TEXT_SECONDARY,
                              FONT_LABEL, FONT_SMALL, FONT_DOT, FONT_MONO, BTN_DANGER)

class ProctorDashboard(tk.Tk):

    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor    # AaimsProcessor — process_frame() called every frame
        self.config    = config       # session metadata dict

        self.processor.alert_callback = self._show_alert
        # wire the alert: aggregator._start_episode() calls this when a flag episode begins
        # this is set HERE (not in aggregator.__init__) because the dashboard is created after processor

        self.title(f"AAIMS — Monitoring: {config.get('student_id', '')} | {config.get('exam_name', '')}")
        self.geometry('780x520')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        # ── Header bar ─────────────────────────────────────────────────────────────────
        header_bar = tk.Frame(self, bg=BG_HEADER, pady=6)
        header_bar.pack(fill='x')    # stretch across full window width
        tk.Label(header_bar,
                 text=f"Student: {config.get('student_id', '')}   |   "
                      f"Exam: {config.get('exam_name', '')}   |   "
                      f"Workstation: {config.get('workstation', '')}",
                 font=FONT_SMALL, bg=BG_HEADER, fg='white').pack()

        # ── Two-column content area ────────────────────────────────────────────────────
        main_content = tk.Frame(self, bg=BG_MAIN)
        main_content.pack(fill='both', expand=True, padx=15, pady=10)

        webcam_panel = tk.Frame(main_content, bg=BG_MAIN)
        webcam_panel.pack(side='left', fill='both', expand=True)
        # left column: webcam feed

        tk.Label(webcam_panel, text='Live Feed', font=FONT_LABEL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(anchor='w')
        self.video_label = tk.Label(webcam_panel, bg='black')
        self.video_label.pack()    # image is drawn here by _update_gui

        status_panel = tk.Frame(main_content, bg=BG_MAIN, width=230)
        status_panel.pack(side='right', fill='y', padx=(12, 0))
        status_panel.pack_propagate(False)
        # width=230 + pack_propagate(False): fixed-width right column regardless of content

        # ── Detection status indicator dots ───────────────────────────────────────────
        tk.Label(status_panel, text='Detection Status',
                 font=FONT_LABEL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(anchor='w', pady=(0, 6))

        self.indicators = {}    # key → Label widget (the colored ● dot)
        for label, key in [
            ('Face Presence',  'face_present'),         # green=detected, red=absent
            ('Head Pose',      'head_flag'),             # green=ok, red=head turned
            ('Face Absence',   'absence_flag'),          # green=ok, red=absent too long
            ('Mouth Movement', 'mouth_alert_active'),    # green=ok, red=talking detected
        ]:
            indicator_row = tk.Frame(status_panel, bg=BG_CARD, pady=6, padx=8)
            indicator_row.pack(fill='x', pady=2)
            tk.Label(indicator_row, text=label, font=FONT_SMALL, bg=BG_CARD, fg=TEXT_PRIMARY).pack(side='left')
            status_dot = tk.Label(indicator_row, text='●', font=FONT_DOT, bg=BG_CARD, fg='green')
            status_dot.pack(side='right')
            self.indicators[key] = status_dot    # store reference for color updates in _update_gui

        ttk.Separator(status_panel, orient='horizontal').pack(fill='x', pady=8)

        # ── Numeric metric labels ──────────────────────────────────────────────────────
        tk.Label(status_panel, text='Metrics', font=FONT_LABEL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(anchor='w')

        self.metric_labels = {}    # key → Label widget (the value text)
        for label, key in [
            ('Direction', 'direction'),
            ('Yaw',       'yaw'),
            ('Pitch',     'pitch'),
            ('MAR',       'mar'),
        ]:
            metric_row = tk.Frame(status_panel, bg=BG_MAIN)
            metric_row.pack(fill='x', pady=1)
            tk.Label(metric_row, text=f'{label}:', font=FONT_SMALL, bg=BG_MAIN,
                     fg=TEXT_SECONDARY, width=9, anchor='w').pack(side='left')
            # width=9: fixed width for label so value column aligns consistently
            metric_value_label = tk.Label(metric_row, text='---', font=FONT_MONO, bg=BG_MAIN, fg=TEXT_PRIMARY)
            metric_value_label.pack(side='left')
            self.metric_labels[key] = metric_value_label

        ttk.Separator(status_panel, orient='horizontal').pack(fill='x', pady=8)

        self.alert_label = tk.Label(status_panel, text='', font=FONT_LABEL,
                                    bg=BG_MAIN, fg='red', wraplength=200, justify='center')
        self.alert_label.pack()    # empty at start; text appears when _show_alert is called

        tk.Button(status_panel, text='End Session', font=FONT_LABEL, **BTN_DANGER, pady=5,
                  command=self._end_session).pack(side='bottom', fill='x', pady=8)
        # side='bottom' pins it to the bottom of the status panel

        # ── Camera and threading setup ─────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)    # open default webcam (index 0)
        self.running = True               # camera loop runs while True
        self._pending_after_ids = []      # tracks all after(5000, _clear_alert) IDs for cleanup
        self._gui_update_id = None        # tracks the latest after(0, _update_gui) ID for cleanup

        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        # daemon=True: thread is killed automatically when the main program exits
        # without daemon, closing the window would not kill the thread → app hangs
        self.thread.start()
        print(f'[Dashboard] Monitoring started — student: {config.get("student_id")} | exam: {config.get("exam_name")}')


    def _camera_loop(self):
        # runs on the background thread — NEVER call Tkinter methods directly here
        while self.running:
            success, frame = self.cap.read()
            # blocking call: waits until the webcam delivers a new frame (~33ms at 30fps)
            if not success:
                continue    # skip if camera read failed (e.g. brief hardware glitch)

            result = self.processor.process_frame(frame)
            # all detection logic runs here, on the background thread

            try:
                self._gui_update_id = self.after(0, self._update_gui, frame, result)
                # schedule _update_gui on the Tkinter main thread at earliest opportunity
                # after(0, ...) = "run as soon as the event loop is free"
                # this is the ONLY thread-safe way to update Tkinter from another thread
                # stores the ID so it can be cancelled if End Session fires first
            except tk.TclError:
                break    # after() itself threw — window was destroyed, exit the loop


    def _update_gui(self, frame, result):
        # runs on the Tkinter main thread (scheduled by _camera_loop via after())
        try:
            if not self.winfo_exists():
                return    # window was destroyed before this callback ran
        except tk.TclError:
            return    # winfo_exists itself threw — Tk application is gone

        # ── Webcam frame display ───────────────────────────────────────────────────────
        rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (480, 360))
        display_image = ImageTk.PhotoImage(Image.fromarray(resized_frame))
        self.video_label.configure(image=display_image)
        self.video_label.image = display_image
        # IMPORTANT: must keep a reference on the label; Python GC would delete the image
        # before Tkinter draws it if only the local variable held it

        # ── Status dot colors ──────────────────────────────────────────────────────────
        for key, status_dot in self.indicators.items():
            current_value = result.get(key, 0)
            if key == 'face_present':
                status_dot.configure(fg='green' if current_value else 'red')
                # face_present logic is inverted: 1 is good (green), 0 is bad (red)
            else:
                status_dot.configure(fg='red' if current_value else 'green')
                # all flags: True=problem (red), False=ok (green)

        # ── Metric labels ──────────────────────────────────────────────────────────────
        self.metric_labels['direction'].configure(text=result.get('direction', '---'))
        self.metric_labels['yaw'].configure(text=f"{result.get('yaw', 0):.1f} deg")
        self.metric_labels['pitch'].configure(text=f"{result.get('pitch', 0):.1f} deg")
        self.metric_labels['mar'].configure(text=f"{result.get('mar', 0):.4f}")


    def _show_alert(self, event_type):
        # called by aggregator._start_episode() — once per flagged episode start
        print(f'[ALERT] {event_type}')
        self.alert_label.configure(text=f'ALERT\n{event_type}')
        after_id = self.after(5000, self._clear_alert)
        # schedule alert clearing 5000ms (5 seconds) from now
        self._pending_after_ids.append(after_id)
        # store ID so it can be cancelled in _end_session before destroy()


    def _clear_alert(self):
        try:
            if self.winfo_exists():
                self.alert_label.configure(text='')    # clear the alert banner text
        except tk.TclError:
            pass    # window already destroyed — nothing to clear


    def _end_session(self):
        print('[Dashboard] Session ended by proctor.')
        self.running = False    # signals _camera_loop to exit on next while check

        if self._gui_update_id is not None:
            try:
                self.after_cancel(self._gui_update_id)
                # cancel any pending _update_gui call that hasn't fired yet
                # prevents "invalid command name" error from a stale callback firing after destroy()
            except Exception:
                pass

        for after_id in self._pending_after_ids:
            try:
                self.after_cancel(after_id)
                # cancel all pending _clear_alert(5s) callbacks
                # without this, each fires against the destroyed Tcl interpreter → error
            except Exception:
                pass

        self.cap.release()    # release webcam hardware back to the OS
        self.destroy()        # destroy Tkinter window and Tcl interpreter

        from frontend.integrity_log_screen import IntegrityLogScreen
        IntegrityLogScreen(self.processor, self.config).mainloop()
        # import is deferred to here to avoid circular imports at module load time
```

---

## 14. Line-by-Line: `frontend/integrity_log_screen.py`

**Purpose:** Post-session log review. Reads the CSV with pandas, displays a summary count and a scrollable episode table, and offers a Desktop export.

```python
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd       # for reading the CSV into a DataFrame
import shutil             # for copying the log file to Desktop
import os
from frontend.styles import (BG_MAIN, TEXT_PRIMARY, TEXT_SECONDARY,
                              FONT_TITLE, FONT_SMALL, BG_FLAGGED,
                              BTN_PRIMARY, BTN_SECONDARY)

class IntegrityLogScreen(tk.Tk):

    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor    # kept for processor.log_path
        self.config    = config       # kept for student_id and exam_name (used in export filename)

        self.title('AAIMS — Session Integrity Log')
        self.geometry('820x500')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        tk.Label(self, text='Post-Examination Integrity Log',
                 font=FONT_TITLE, bg=BG_MAIN, fg=TEXT_PRIMARY).pack(pady=(18, 2))

        tk.Label(self,
                 text=f"Student: {config.get('student_id', '')}   |   Exam: {config.get('exam_name', '')}",
                 font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack()

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30, pady=10)

        self.summary_label = tk.Label(self, text='Loading...', font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_PRIMARY)
        self.summary_label.pack(pady=(0, 8))
        # text is replaced with actual counts once _load_log() runs

        # ── Scrollable episode table ───────────────────────────────────────────────────
        table_frame = tk.Frame(self)
        table_frame.pack(fill='both', expand=True, padx=30, pady=(0, 10))

        cols = ('Start Time', 'End Time', 'Duration', 'Description')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=12)
        # show='headings': hides the default leftmost tree column (which is empty)
        # height=12: shows 12 rows before scrolling

        self.tree.heading('Start Time',  text='Start Time')
        self.tree.heading('End Time',    text='End Time')
        self.tree.heading('Duration',    text='Duration')
        self.tree.heading('Description', text='Description')

        self.tree.column('Start Time',  width=145, anchor='center')
        self.tree.column('End Time',    width=145, anchor='center')
        self.tree.column('Duration',    width=80,  anchor='center')
        self.tree.column('Description', width=370, anchor='w')    # left-aligned, widest column

        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        # link scrollbar and treeview: scrollbar controls treeview, treeview reports position to scrollbar
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.tree.tag_configure('flagged', background=BG_FLAGGED)
        # define a tag; rows inserted with tags=('flagged',) get pink (#FFE0E0) background

        # ── Export and Close buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG_MAIN)
        btn_frame.pack(fill='x', padx=30, pady=8)

        tk.Button(btn_frame, text='Export Log to Desktop',
                  font=FONT_SMALL, **BTN_PRIMARY, pady=5, padx=12,
                  command=self._export).pack(side='left')

        tk.Button(btn_frame, text='Close',
                  font=FONT_SMALL, **BTN_SECONDARY, pady=5, padx=12,
                  command=self.destroy).pack(side='right')

        self._load_log()    # populate the table immediately on construction


    def _load_log(self):
        try:
            session_log = pd.read_csv(self.processor.log_path)
            # reads e.g. logs/session_2026-05-26.csv into a pandas DataFrame
            # columns: start_time, end_time, duration_seconds, event_type, description

            if session_log.empty:
                self.summary_label.configure(text='No behavioral events recorded during this session.')
                return

            total_episodes       = len(session_log)
            head_turn_count      = len(session_log[session_log['event_type'] == 'HEAD_TURN'])
            face_absent_count    = len(session_log[session_log['event_type'] == 'FACE_ABSENT'])
            mouth_movement_count = len(session_log[session_log['event_type'] == 'MOUTH_OPEN'])
            # boolean indexing: session_log['event_type'] == 'HEAD_TURN' returns a boolean Series
            # session_log[bool_series] filters rows; len() counts them

            self.summary_label.configure(
                text=f'Total Episodes: {total_episodes}   |   '
                     f'Head Turned: {head_turn_count}   |   '
                     f'Face Absent: {face_absent_count}   |   '
                     f'Mouth Movement: {mouth_movement_count}')

            for _, episode in session_log.iterrows():
                # iterrows() yields (row_index, Series) pairs; _ discards the index
                self.tree.insert(
                    '', 'end',            # '' = insert at root level; 'end' = at the bottom
                    values=(
                        episode['start_time'],
                        episode['end_time'],
                        f"{episode['duration_seconds']}s",    # e.g. "4s"
                        episode['description']
                    ),
                    tags=('flagged',))    # applies the pink background defined by tag_configure

        except FileNotFoundError:
            self.summary_label.configure(text='No events logged for this session.')
        except Exception as e:
            self.summary_label.configure(text=f'Error loading log: {e}')


    def _export(self):
        try:
            desktop_directory = os.path.join(os.path.expanduser('~'), 'Desktop')
            # expanduser('~') → C:\Users\<username> on Windows
            # os.path.join appends 'Desktop' with the correct separator

            filename = (
                f"AAIMS_Log_"
                f"{self.config.get('student_id', 'unknown')}_"
                f"{self.config.get('exam_name', '')}.csv"
            ).replace(' ', '_')
            # replace spaces so the filename is safe across OS file browsers

            destination_path = os.path.join(desktop_directory, filename)
            shutil.copy(self.processor.log_path, destination_path)
            # byte-for-byte copy; original log at logs/ is not moved or deleted

            messagebox.showinfo('Export Successful', f'Log exported to Desktop:\n{filename}')
        except Exception as e:
            messagebox.showerror('Export Failed', str(e))
```

---

## 15. Key Algorithms Explained

### Mouth Aspect Ratio (MAR)

```
MAR = |upper_lip - lower_lip| / (|left_corner - right_corner| + ε)
```

Dividing by horizontal width normalizes for face size and distance from camera. A person far from the camera has a smaller face, so the absolute pixel distance would be small even with an open mouth — dividing by width cancels this out.

---

### PnP (Perspective-n-Point) Problem

Given:
- N points with known 3D coordinates
- Their corresponding 2D projections in the image
- The camera intrinsic matrix

Find: the rotation (R) and translation (T) that maps 3D → 2D.

OpenCV's `solvePnP` uses the Efficient PnP (EPnP) algorithm by default. The solution is a 3×1 rotation vector (axis-angle / Rodrigues form) and a 3×1 translation vector.

---

### Rodrigues Rotation

The rotation vector from `solvePnP` is in axis-angle form: the direction is the rotation axis and the magnitude is the angle in radians. `cv2.Rodrigues()` converts this to a 3×3 rotation matrix:

```
R = I·cos(θ) + (1-cos(θ))·k·kᵀ + sin(θ)·[k]ₓ
```

where θ is the rotation angle and k is the unit axis vector.

---

### RQ Decomposition

`cv2.RQDecomp3x3()` decomposes a 3×3 rotation matrix into Euler angles. The output values are near 0.0 due to the input scale, so they are multiplied by 360 to scale into practical degree ranges (approximately ±30° for typical head poses).

---

### Support Vector Machine (SVM)

An SVM finds the hyperplane that maximally separates two classes. For the mouth classifier, the feature space is one-dimensional (scaled MAR). The SVM finds the threshold on the scaled MAR axis that best separates `closed` from `open` samples with maximum margin.

`StandardScaler` preprocessing is critical: SVMs assume features are on similar scales. Standardizing to zero mean and unit variance ensures the MAR values sit in the expected range of the model.

---

### Haar Cascade

A cascade classifier made up of many simple Haar-feature classifiers arranged in stages. A region is rejected as soon as it fails any stage, making detection fast. Features measure differences between rectangular pixel intensity sums — e.g., the eye region is typically darker than the cheek region.

**Limitation:** Trained on upright frontal faces. Significant head tilt rotates the face away from the frontal template, causing misses. This is why AAIMS uses MediaPipe (which handles tilts) for face_present, and uses Haar only to confirm presence for the absence counter.

---

## 16. Event Logging System

Log file location: `logs/session_YYYY-MM-DD.csv`

Created fresh at startup. The `AaimsProcessor.__init__` method creates the file — starting a new session overwrites today's file.

### CSV Schema

| Column | Type | Description |
|---|---|---|
| `start_time` | datetime string | When the flagged behavior began |
| `end_time` | datetime string | When the flagged behavior ended |
| `duration_seconds` | int | How long the episode lasted |
| `event_type` | string | `HEAD_TURN`, `FACE_ABSENT`, or `MOUTH_OPEN` |
| `description` | string | Human-readable label |

### Episode vs. Frame Logging

A naive approach would log one row per flagged frame (thousands of rows per session). AAIMS logs **episodes** — one row per continuous stretch of the same flag. This makes the log meaningful to a human reviewer.

The `active_episodes` dict tracks which events are currently active:

```python
self.active_episodes = {
    'HEAD_TURN':   None,   # None = not active; datetime = start of active episode
    'FACE_ABSENT': None,
    'MOUTH_OPEN':  None,
}
```

**Known limitation:** If a flag is still active when End Session is clicked, that episode is never closed. The partial episode is not written to the CSV — only cleanly ended episodes are logged.

---

## 17. Design Decisions and Trade-offs

### Why SVM for mouth detection instead of a threshold on MAR?

A fixed MAR threshold would not generalize across people (wider mouths, different camera distances). The SVM learns the boundary from actual data. The StandardScaler normalizes for inter-person variability in absolute MAR values.

### Why the calibration MAR baseline gate alongside the SVM?

The SVM's decision boundary is frozen from training on one person. A student sitting closer or with different facial proportions may have a resting MAR above the SVM's closed/open boundary, causing persistent false positives. The 1.5× gate rejects those. If calibration was skipped (`mar_baseline == 0.0`), the gate is bypassed and SVM decides alone.

### Why is `is_talking` a sustained state rather than a one-shot signal?

If `detect_talking` returned True for exactly one frame, `_start_episode` and `_end_episode` would fire on back-to-back frames, always logging `duration=0`. The sustained `is_talking` state (with 30-frame inactivity decay) keeps `mouth_flag` True for the full talking episode so the logged duration is meaningful.

### Why fire the alert in `_start_episode` instead of every flagged frame?

At 30fps, a 10-second flag produces 300 flagged frames. Firing `alert_callback` every frame queues 300 `after(5000, _clear_alert)` callbacks. When the session ends, all 300 fire against the destroyed Tk interpreter. Moving the call to `_start_episode` (once per episode) keeps the queue size small and the cleanup in `_end_session` manageable.

### Why AND logic for the absence counter?

- Haar alone: misses tilted faces; has false positives on background face-like patterns
- MediaPipe alone: occasional single-frame flickering re-detections reset the counter prematurely

AND logic eliminates both failure modes. Counter resets only when there is strong bilateral evidence the face is present.

### Why calibrate instead of fixed thresholds?

Students sit at different distances and angles from the webcam. A 10-degree leftward yaw might be one student's straight-ahead position. Calibration captures each student's individual neutral pose so all detection is relative to it.

### Why a background thread for the camera loop?

`cv2.VideoCapture.read()` blocks until the camera delivers a frame. Calling it on the Tkinter main thread would freeze the UI entirely. The background thread runs the camera loop continuously while Tkinter's mainloop handles UI events normally. Cross-thread communication uses `self.after(0, ...)` — the only thread-safe way to schedule code on the Tkinter main thread.

### Why `_gui_update_id` tracking?

When End Session is clicked, `self.running = False` stops the camera loop from scheduling new `after(0, _update_gui, ...)` calls. But there is one already-queued callback that fires after `destroy()` — this produces `invalid command name` errors. Cancelling `_gui_update_id` before `destroy()` eliminates that last callback.
