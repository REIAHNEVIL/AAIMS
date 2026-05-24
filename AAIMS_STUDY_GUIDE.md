# AAIMS — Automated Academic Integrity Monitoring System
## Comprehensive Study Guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [File Structure](#3-file-structure)
4. [Application Flow](#4-application-flow)
5. [Detection Modules](#5-detection-modules)
   - 5.1 [Face Absence Detection](#51-face-absence-detection--modulesface_absencepy)
   - 5.2 [Head Pose Detection](#52-head-pose-detection--moduleshead_posepy)
   - 5.3 [Mouth Movement Detection](#53-mouth-movement-detection--modulesmouth_detectionpy)
6. [The Aggregator](#6-the-aggregator--modulesaggregatorpy)
7. [GUI Screens](#7-gui-screens)
   - 7.1 [Entry Point](#71-entry-point--mainpy)
   - 7.2 [Notice Screen](#72-notice-screen--guinotice_screenpy)
   - 7.3 [Session Setup](#73-session-setup--guisession_setuppy)
   - 7.4 [Calibration Screen](#74-calibration-screen--guicalibration_screenpy)
   - 7.5 [Proctor Dashboard](#75-proctor-dashboard--guiproctor_dashboardpy)
   - 7.6 [Integrity Log Screen](#76-integrity-log-screen--guiintegrity_log_screenpy)
8. [Data Collection Scripts](#8-data-collection-scripts)
9. [Key Algorithms Explained](#9-key-algorithms-explained)
10. [Event Logging System](#10-event-logging-system)
11. [Design Decisions and Trade-offs](#11-design-decisions-and-trade-offs)

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

## 2. System Architecture

```
Webcam Frame (BGR)
       │
       ├──► Grayscale ──► FaceAbsenceDetector.predict()   (Haar Cascade)
       │                         │
       └──► RGB ──► MediaPipe FaceMesh ──► face_landmarks
                          │
              ┌───────────┼────────────┐
              │           │            │
         HeadPose    MouthDetection   face_present
         get_angles   compute_mar     (bool)
         solvePnP     SVM classify
              │           │
         pitch/yaw/roll  talking_detected
              │
         detect_head_turned()
              │
         head_flag
                    │
            AaimsProcessor.process_frame()
                    │
            ┌───────┴────────────────┐
            │  _update_episodes()    │
            │  episode log (CSV)     │
            └───────┬────────────────┘
                    │
            Result dict ──► ProctorDashboard GUI
```

### Two-Detector Face Presence Logic

The absence counter uses **both** Haar Cascade and MediaPipe together:

```
Reset counter (face present): Haar AND MediaPipe both detect face
Increment counter (possible absence): either one misses
```

This prevents:
- Haar false positives (background patterns keeping counter at 0)
- MediaPipe flickering (single-frame re-detections resetting counter)

---

## 3. File Structure

```
AAIMS/
│
├── main.py                          ← Entry point, launches GUI chain
│
├── modules/
│   ├── face_absence.py              ← Haar Cascade face presence detector
│   ├── head_pose.py                 ← solvePnP head angle estimator
│   ├── mouth_detection.py           ← SVM-based mouth open/closed classifier
│   └── aggregator.py               ← Combines all three, manages logs
│
├── gui/
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
├── data/
│   ├── face_dataset/
│   │   └── collect_data.py          ← Collects face/no_face images
│   ├── mouth_dataset/
│   │   └── collect_data.py          ← Collects MAR values with labels
│   └── behavior_dataset/
│       └── collect_behavior.py      ← (Planned) behavioral feature CSV
│
├── logs/
│   └── session_YYYY-MM-DD.csv       ← Auto-generated event log per day
│
└── headPoseEstimation.py            ← Standalone reference/tutorial script
```

---

## 4. Application Flow

The application runs as a sequential chain of Tkinter windows. Each window calls the next via a callback passed at construction time.

```
main.py
  │
  └─► NoticeScreen(on_acknowledge=launch_session_setup)
            │ student clicks "I Acknowledge"
            ▼
      SessionSetupScreen(on_start=launch_calibration)
            │ proctor fills form, clicks Start
            ▼
      CalibrationScreen(processor, config, on_done=launch_dashboard)
            │ 100 frames collected, baselines computed
            ▼
      ProctorDashboard(processor, config)
            │ proctor clicks End Session
            ▼
      IntegrityLogScreen(processor, config)
```

The `AaimsProcessor` is instantiated **once** in `main.py` and passed through every screen. This means models are loaded once at startup and the same processor, log file, and detection state persist for the whole exam session.

---

## 5. Detection Modules

### 5.1 Face Absence Detection — `modules/face_absence.py`

**Method:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)

Haar Cascade is a classical machine learning method (not deep learning). It scans the image at multiple scales looking for regions that match a learned set of Haar-like features (rectangular pixel intensity patterns). It is fast and reliable for frontal faces, but degrades on tilted or rotated faces.

```python
class FaceAbsenceDetector:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.absence_counter    = 0
        self.ABSENCE_THRESHOLD  = 30    # ~1 second at 30fps
        self.current_face_state = 1     # 1=present, 0=absent
```

**`predict(grayscale_frame)`** — runs the Haar Cascade and returns 1 (face found) or 0 (not found).

```python
    def predict(self, grayscale_frame):
        faces = self.face_cascade.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.05,    # how much image is shrunk per scale
            minNeighbors=4,      # how many neighbors needed to keep a detection
            minSize=(30, 30)     # minimum face size in pixels
        )
        return 1 if len(faces) > 0 else 0
```

**Parameters explained:**
- `scaleFactor=1.05` — the image is shrunk by 5% at each scale. Lower = more thorough but slower.
- `minNeighbors=4` — a detection rectangle must have at least 4 overlapping neighbors to be accepted. Higher = fewer false positives.
- `minSize=(30, 30)` — ignores anything smaller than 30×30 pixels.

**`detect_absence(grayscale_frame)`** — updates the running counter and returns True when the threshold is crossed.

```python
    def detect_absence(self, grayscale_frame):
        self.current_face_state = self.predict(grayscale_frame)
        if self.current_face_state == 0:
            self.absence_counter += 1
        else:
            self.absence_counter = 0
        return self.absence_counter >= self.ABSENCE_THRESHOLD
```

> **Note:** `detect_absence()` is not called by the aggregator. The aggregator calls `predict()` directly so it can combine Haar's result with MediaPipe's result before deciding whether to increment the counter.

---

### 5.2 Head Pose Detection — `modules/head_pose.py`

**Method:** MediaPipe Face Mesh landmarks → OpenCV `solvePnP` → Euler angles

This module does not use a trained model. It is a geometric computation: given 6 known 3D face landmark positions and their 2D projections in the image, it solves for the camera-to-face rotation and translation.

#### Landmark Indices Used

| Index | Landmark |
|---|---|
| 1 | Nose tip |
| 33 | Left eye outer corner |
| 61 | Left mouth corner |
| 199 | Chin |
| 263 | Right eye outer corner |
| 291 | Right mouth corner |

These 6 points are chosen because they form a stable, spread-out configuration that gives reliable PnP solutions.

#### `get_angles()` — Step by Step

```python
def get_angles(self, face_landmarks, image_width, image_height):
```

**Step 1: Extract 2D and 3D coordinates**

```python
    for landmark_index, landmark in enumerate(face_landmarks.landmark):
        if landmark_index in [1, 33, 61, 199, 263, 291]:
            x_pixel = int(landmark.x * image_width)
            y_pixel = int(landmark.y * image_height)
            face_2d.append([x_pixel, y_pixel])
            face_3d.append([x_pixel, y_pixel, landmark.z])
```

MediaPipe returns normalized coordinates (0.0–1.0). Multiplying by image dimensions gives pixel coordinates. The `landmark.z` value is relative depth — it is used directly as a pseudo-3D coordinate.

**Step 2: Build the camera matrix**

```python
    focal_length  = 1 * image_width
    camera_matrix = np.array([
        [focal_length, 0,            image_height / 2],
        [0,            focal_length, image_width  / 2],
        [0,            0,            1                ]
    ])
```

This is a simplified pinhole camera model. The focal length is approximated as equal to the image width, which is a common shortcut when no real camera calibration data is available. The principal point (optical center) is assumed to be the image center.

**Step 3: Solve PnP**

```python
    solved, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d, face_2d, camera_matrix, distortion_coefficients
    )
```

`solvePnP` solves the Perspective-n-Point problem: given the 3D points and their 2D projections, find the rotation and translation that maps 3D→2D. The output `rotation_vector` is a 3×1 vector in axis-angle form (Rodrigues representation).

**Step 4: Convert to Euler angles**

```python
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = euler_angles[0] * 360
    yaw   = euler_angles[1] * 360
    roll  = euler_angles[2] * 360
```

`cv2.Rodrigues()` converts the axis-angle vector to a 3×3 rotation matrix.
`cv2.RQDecomp3x3()` decomposes the rotation matrix into Euler angles via RQ decomposition. The raw angles are very small (near 0), so multiplying by 360 scales them into a usable degree range.

**Angle conventions:**
- **Pitch** (x-axis rotation): positive = looking up, negative = looking down
- **Yaw** (y-axis rotation): positive = looking left, negative = looking right
- **Roll** (z-axis rotation): head tilt, not used for flagging

#### `calibrate()` — Baseline Computation

```python
def calibrate(self, angle_buffer):
    self.pitch_baseline = np.mean([a[0] for a in angle_buffer])
    self.yaw_baseline   = np.mean([a[1] for a in angle_buffer])
```

During calibration, 100 frames of pitch/yaw pairs are collected while the student sits naturally. The mean is stored as the neutral baseline. All subsequent detections compare against this baseline, accounting for different seating positions and camera angles.

#### `detect_head_turned()` — Flag Logic

```python
def detect_head_turned(self, pitch, yaw):
    pitch_deviation = pitch - self.pitch_baseline
    yaw_deviation   = abs(yaw - self.yaw_baseline)

    horizontal_turn = yaw_deviation   > self.yaw_threshold    # 10 degrees
    looking_down    = pitch_deviation < -self.pitch_threshold  # 10 degrees

    if horizontal_turn or looking_down:
        self.sustained_off_screen_frames += 1
    else:
        self.sustained_off_screen_frames = 0

    return self.sustained_off_screen_frames >= self.FRAMES_TO_FLAG_HEAD_TURN  # 20 frames
```

**Why looking up is excluded:** The classroom has student assistants monitoring from above, so a student briefly looking up is not a concern. The exam is on-screen, so left, right, and downward movement are the suspicious directions.

**Why the threshold (20 frames)?** At ~30fps, 20 frames ≈ 0.67 seconds. A brief natural glance does not trigger the flag; a sustained look away does.

---

### 5.3 Mouth Movement Detection — `modules/mouth_detection.py`

**Method:** Geometric MAR computation → StandardScaler → SVM classifier → open/close cycle counting

This module uses a custom-trained Support Vector Machine model. The model was trained on MAR (Mouth Aspect Ratio) values collected with `data/mouth_dataset/collect_data.py`.

#### Mouth Aspect Ratio (MAR)

MAR is a single number that describes how open the mouth is:

```
MAR = vertical lip distance / horizontal mouth width
```

A closed mouth has a MAR near 0. An open mouth has a higher MAR.

```python
UPPER_LIP    = 13   # MediaPipe landmark index
LOWER_LIP    = 14
LEFT_CORNER  = 61
RIGHT_CORNER = 291

def compute_mar(self, landmarks, image_width, image_height):
    def landmark_to_pixel(landmark_index):
        landmark = landmarks[landmark_index]
        return np.array([landmark.x * image_width, landmark.y * image_height])

    vertical_lip_distance  = np.linalg.norm(
        landmark_to_pixel(UPPER_LIP) - landmark_to_pixel(LOWER_LIP))
    horizontal_mouth_width = np.linalg.norm(
        landmark_to_pixel(LEFT_CORNER) - landmark_to_pixel(RIGHT_CORNER))

    return vertical_lip_distance / (horizontal_mouth_width + 1e-6)
```

The `+ 1e-6` prevents division by zero on degenerate frames where the horizontal width is zero.

#### SVM Classification

```python
def detect_talking(self, landmarks, image_width, image_height):
    mouth_aspect_ratio = self.compute_mar(landmarks, image_width, image_height)
    self.last_mar      = mouth_aspect_ratio

    normalized_mar           = self.feature_scaler.transform([[mouth_aspect_ratio]])
    self.current_mouth_state = self.mouth_classifier.predict(normalized_mar)[0]
```

The raw MAR is first scaled using the `StandardScaler` (subtracts the training mean, divides by training standard deviation). The scaled value is then passed to the SVM which outputs 0 (closed) or 1 (open).

**Why scale first?** SVMs are sensitive to feature magnitude. The scaler ensures the MAR values are in the same range as what the model was trained on.

#### Open-Close Cycle Counting

Simply being open is not enough to flag — a yawn also opens the mouth. AAIMS counts **cycles** (open then close) and requires multiple cycles, which is the pattern of talking:

```python
    talking_detected = False

    if self.current_mouth_state == 1:
        self.open_frame_count += 1
    else:
        if self.open_frame_count >= self.MIN_OPEN_FRAMES_PER_CYCLE:  # 5 frames
            self.open_close_cycle_count += 1

            if self.open_close_cycle_count >= self.CYCLES_TO_FLAG_TALKING:  # 3 cycles
                talking_detected            = True
                self.open_close_cycle_count = 0

        self.open_frame_count = 0

    return talking_detected
```

**Logic walkthrough:**
1. While mouth is open: increment `open_frame_count`
2. When mouth closes: if it was open for at least 5 frames, that counts as one complete cycle
3. After 3 complete cycles: flag talking, reset cycle counter so the next episode can be detected

**Why 5 frames minimum per cycle?** Prevents a single-frame noise detection from counting as a cycle.
**Why 3 cycles?** A yawn is one slow open-close. Talking produces rapid repeated cycles.

---

## 6. The Aggregator — `modules/aggregator.py`

`AaimsProcessor` is the central coordinator. It owns all three detectors, the MediaPipe FaceMesh instance, and the event log.

### `process_frame(frame)` — Complete Logic

Called once per webcam frame by the dashboard's background thread.

```python
def process_frame(self, frame):
    image_height, image_width = frame.shape[:2]

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

Two color conversions: grayscale for Haar Cascade (only needs intensity), RGB for MediaPipe (expects RGB, not BGR).

```python
    results                = self.face_mesh.process(rgb_frame)
    mediapipe_detects_face = bool(results.multi_face_landmarks)
    haar_detects_face      = self.face_absence.predict(grayscale_frame) == 1

    if haar_detects_face and mediapipe_detects_face:
        self.face_absence.absence_counter = 0
    else:
        self.face_absence.absence_counter += 1
```

**AND logic for the absence counter:** Both detectors must agree the face is present to reset the counter. This prevents:
- Haar false positives (posters, background patterns) from blocking absence detection
- MediaPipe single-frame flickers from resetting the counter prematurely

```python
    if mediapipe_detects_face:
        face_present = 1
        landmarks    = results.multi_face_landmarks[0]

        pitch, yaw, roll, *_ = self.head_pose.get_angles(
            landmarks, image_width, image_height)

        if pitch is not None:
            direction = self.head_pose.get_direction(pitch, yaw)
            head_flag = self.head_pose.detect_head_turned(pitch, yaw)

        mouth_flag         = self.mouth.detect_talking(
            landmarks.landmark, image_width, image_height)
        mouth_aspect_ratio = self.mouth.last_mar
```

**MediaPipe drives face_present and landmarks:** MediaPipe handles tilted faces that Haar misses. Head pose and mouth detection only run when MediaPipe has landmarks to work with.

```python
    absence_flag = (
        self.face_absence.absence_counter >=
        self.face_absence.ABSENCE_THRESHOLD
    )
```

`absence_flag` is distinct from `face_present`. `face_present` is immediate (one frame). `absence_flag` requires the counter to reach 30 consecutive non-confirmed frames.

### Event Episode System

Instead of logging every individual flagged frame, AAIMS logs **episodes** — continuous stretches of flagged behavior with start time, end time, and duration.

```python
def _start_episode(self, event_type):
    if self.active_episodes[event_type] is None:
        self.active_episodes[event_type] = datetime.datetime.now()

def _end_episode(self, event_type):
    start    = self.active_episodes[event_type]
    end      = datetime.datetime.now()
    duration = (end - start).seconds

    with open(self.log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([start, end, duration, event_type, description])

    self.active_episodes[event_type] = None
```

`_update_episodes()` is called every frame. When a flag turns on, `_start_episode` records the start time. When it turns off, `_end_episode` writes the row and clears the record.

### Return Dictionary

`process_frame` returns a dict consumed by the GUI:

```python
return {
    'pitch':              pitch,           # float, degrees
    'yaw':                yaw,             # float, degrees
    'roll':               roll,            # float, degrees
    'mar':                mouth_aspect_ratio,
    'face_present':       face_present,    # 0 or 1
    'direction':          direction,       # string label
    'head_flag':          head_flag,       # bool
    'absence_flag':       absence_flag,    # bool
    'mouth_flag':         mouth_flag,      # bool (one-shot per episode)
    'mouth_alert_active': mouth_alert_active,  # bool (5-second timer)
    'flagged':            len(flagged_events) > 0
}
```

---

## 7. GUI Screens

### 7.1 Entry Point — `main.py`

```python
processor = AaimsProcessor()   # load models once

def launch_notice():
    NoticeScreen(on_acknowledge=launch_session_setup).mainloop()

def launch_session_setup():
    SessionSetupScreen(on_start=launch_calibration).mainloop()

def launch_calibration(config):
    CalibrationScreen(processor, config, on_done=launch_dashboard).mainloop()

def launch_dashboard(config):
    ProctorDashboard(processor, config).mainloop()

if __name__ == '__main__':
    launch_notice()
```

Each function is a callback passed to the previous screen. When a screen finishes, it calls its callback which launches the next screen. The `processor` object travels through the whole chain.

---

### 7.2 Notice Screen — `gui/notice_screen.py`

Displays the consent notice before the exam begins. No detection logic — purely informational UI. The "I Acknowledge" button calls `self.destroy()` then `self.on_acknowledge()`, which chains to session setup.

---

### 7.3 Session Setup — `gui/session_setup.py`

Collects three fields from the proctor: student ID, exam name, workstation number. These are packaged into a `config` dict:

```python
config = {
    'student_id':        self.student_id.get().strip(),
    'exam_name':         self.exam_name.get().strip(),
    'workstation':       self.workstation.get().strip(),
    'yaw_threshold':     10,
    'absence_threshold': 30
}
```

The `config` dict is passed to every subsequent screen and used to label the log file and dashboard title.

---

### 7.4 Calibration Screen — `gui/calibration_screen.py`

Captures 100 frames of the student sitting in a neutral position to establish personalized baselines.

**Why calibrate?** Students sit at different distances and angles from the camera. Without calibration, a student sitting slightly left of center would appear to always be "looking left." The baseline captures their natural resting position and all subsequent detection is relative to it.

**What is collected:**

```python
def _process_calibration_frame(self, frame):
    angles = self.processor.head_pose.get_angles(landmarks, w, h)
    if angles[0] is not None:
        self.angle_buffer.append((angles[0], angles[1]))  # (pitch, yaw)

    mouth_aspect_ratio = self.processor.mouth.compute_mar(landmarks.landmark, w, h)
    self.mar_buffer.append(mouth_aspect_ratio)
```

**What is computed from the buffer:**

```python
def _finish(self):
    self.processor.head_pose.calibrate(self.angle_buffer)
    # sets pitch_baseline = mean(all pitch values)
    # sets yaw_baseline   = mean(all yaw values)

    self.processor.mouth.mar_baseline = float(np.mean(self.mar_buffer))
    # stores the resting MAR (mouth closed baseline)
```

**Tkinter threading note:** The webcam preview uses `self.after(33, self._update_preview)` — a 33ms recursive callback that keeps the UI live. The `WM_DELETE_WINDOW` protocol handler ensures that if the user closes the window early, the camera is released and `self.running` is set to False before destroy, preventing orphaned callbacks.

---

### 7.5 Proctor Dashboard — `gui/proctor_dashboard.py`

The main monitoring screen. Uses a **background thread** for the camera loop to avoid blocking the Tkinter mainloop.

```python
self.thread = threading.Thread(target=self._camera_loop, daemon=True)
self.thread.start()
```

**Why a thread?** `cap.read()` blocks until a frame is available. If called on the main thread, it would freeze the UI. The daemon flag means the thread is killed automatically when the main program exits.

```python
def _camera_loop(self):
    while self.running:
        success, frame = self.cap.read()
        if not success:
            continue
        result = self.processor.process_frame(frame)
        try:
            self.after(0, self._update_gui, frame, result)
        except tk.TclError:
            break
```

`self.after(0, ...)` schedules `_update_gui` to run on the main thread at the next opportunity. This is the safe Tkinter pattern for cross-thread GUI updates. The `try/except TclError` catches the race condition where the window is destroyed between the `while self.running` check and the `after()` call.

```python
def _update_gui(self, frame, result):
    if not self.winfo_exists():
        return
    ...
```

The `winfo_exists()` guard handles any callbacks that arrived in the queue after `destroy()` was called.

**Status indicators** use a `●` character colored green or red:

```python
for key, status_dot in self.indicators.items():
    current_value = result.get(key, 0)
    if key == 'face_present':
        status_dot.configure(fg='green' if current_value else 'red')
    else:
        status_dot.configure(fg='red' if current_value else 'green')
```

`face_present` is green when 1 (face found), red when 0. All other indicators (head_flag, absence_flag, mouth_alert_active) are green when 0 (no issue) and red when 1 (flagged).

**Alert display timer:** Mouth movement is a one-shot event (returns True for a single frame). To keep it visible on screen, a timestamp is recorded and the alert stays shown for 5 seconds:

```python
if mouth_flag:
    self.mouth_alert_time = time.time()

mouth_alert_active = (time.time() - self.mouth_alert_time) < self.MOUTH_ALERT_DISPLAY
```

---

### 7.6 Integrity Log Screen — `gui/integrity_log_screen.py`

Displayed after the session ends. Reads the CSV log with `pandas.read_csv()` and populates a `ttk.Treeview` table.

```python
session_log          = pd.read_csv(self.processor.log_path)
total_episodes       = len(session_log)
head_turn_count      = len(session_log[session_log['event_type'] == 'HEAD_TURN'])
face_absent_count    = len(session_log[session_log['event_type'] == 'FACE_ABSENT'])
mouth_movement_count = len(session_log[session_log['event_type'] == 'MOUTH_OPEN'])
```

The export button copies the CSV to the proctor's Desktop with a filename that includes the student ID and exam name.

---

## 8. Data Collection Scripts

### `data/face_dataset/collect_data.py`

Collects grayscale 128×128 face images for training a face presence classifier. Run once with `LABEL = 'face'` and once with `LABEL = 'no_face'` to build a balanced dataset.

```python
LABEL              = 'no_face'     # change to 'face' for the other class
SAVE_DIRECTORY     = f'data/face_dataset/{LABEL}'
TARGET_FRAME_COUNT = 900

if time.time() - last_save >= 0.3:
    gray_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (128, 128))
    cv2.imwrite(filename, resized_frame)
```

Saves one image every 0.3 seconds to avoid near-duplicate frames.

---

### `data/mouth_dataset/collect_data.py`

Collects MAR values labeled as open (1) or closed (0) into a CSV. This CSV is used directly to train the SVM.

```python
LABEL               = 1            # 0 = closed, 1 = open
TARGET_SAMPLE_COUNT = 500
CSV_FILE            = 'data/mouth_dataset/mar_data.csv'

mouth_aspect_ratio = compute_mar(face_landmarks, frame_width, frame_height)
writer.writerow([mouth_aspect_ratio, LABEL])
```

Run once with `LABEL = 0` (mouth closed) and once with `LABEL = 1` (mouth open). The resulting CSV has two columns: `mar_value, label`.

---

## 9. Key Algorithms Explained

### Mouth Aspect Ratio (MAR)

```
MAR = |upper_lip - lower_lip| / (|left_corner - right_corner| + ε)
```

Dividing by horizontal width normalizes for face size and distance from camera. A person far from the camera has a smaller face, so the absolute pixel distance would be small even if their mouth is wide open — dividing by the mouth width cancels this out.

---

### PnP (Perspective-n-Point) Problem

Given:
- N points with known 3D world coordinates
- Their corresponding 2D projections in the image
- The camera intrinsic matrix

Find: the rotation (R) and translation (T) that best explains the 2D projections.

OpenCV's `solvePnP` uses the Efficient PnP (EPnP) algorithm by default. The solution is a 3×1 rotation vector (axis-angle) and a 3×1 translation vector.

---

### Rodrigues Rotation

The rotation vector from `solvePnP` is in axis-angle form: the direction of the vector is the rotation axis and the magnitude is the rotation angle in radians. `cv2.Rodrigues()` converts this to a 3×3 rotation matrix using the Rodrigues formula:

```
R = I·cos(θ) + (1-cos(θ))·k·kᵀ + sin(θ)·[k]ₓ
```

where θ is the rotation angle and k is the unit axis vector.

---

### RQ Decomposition

`cv2.RQDecomp3x3()` decomposes a rotation matrix into Euler angles. RQ decomposition factors a matrix as R (upper triangular) × Q (orthogonal). The three angles represent pitch, yaw, and roll in the camera's coordinate frame.

The raw output angles are multiplied by 360 to scale them into a practical range of approximately ±30 degrees for typical head poses.

---

### Support Vector Machine (SVM)

An SVM finds the hyperplane that maximally separates two classes. For the mouth classifier, the feature space is one-dimensional (MAR values after scaling). The SVM finds a threshold on the scaled MAR axis that best separates `closed` samples from `open` samples, with maximum margin.

The `StandardScaler` preprocessing is important: SVMs assume features are on similar scales. Standardizing to zero mean and unit variance ensures the single MAR feature sits in the expected range of the model.

---

### Haar Cascade

A cascade classifier made up of many simple Haar-feature classifiers arranged in stages. A region is rejected as soon as it fails any stage, making detection very fast. Features look like differences between rectangular pixel intensity sums — for example, the eye region is typically darker than the cheek region.

**Limitation:** Trained on upright frontal faces. Significant head tilt rotates the face away from the frontal template, causing misses. This is why AAIMS uses MediaPipe (which handles tilts) for the face_present indicator and only uses Haar to confirm presence for the absence counter.

---

## 10. Event Logging System

Log file location: `logs/session_YYYY-MM-DD.csv`

Created fresh at startup by `_init_log()`. Multiple sessions on the same day append to the same file (the log is initialized at `AaimsProcessor.__init__` time, so starting a new `AaimsProcessor` overwrites the file).

### CSV Schema

| Column | Type | Description |
|---|---|---|
| `start_time` | datetime string | When the flagged behavior began |
| `end_time` | datetime string | When the flagged behavior ended |
| `duration_seconds` | int | How long the episode lasted |
| `event_type` | string | `HEAD_TURN`, `FACE_ABSENT`, or `MOUTH_OPEN` |
| `description` | string | Human-readable label |

### Episode vs. Frame Logging

A naive approach would log one row per flagged frame (thousands of rows per session). AAIMS uses the episode approach: one row per continuous stretch of the same flag. This makes the log readable and meaningful to a human reviewer.

The `active_episodes` dict tracks which events are currently active:

```python
self.active_episodes = {
    'HEAD_TURN':   None,   # None = not active; datetime = start of active episode
    'FACE_ABSENT': None,
    'MOUTH_OPEN':  None,
}
```

---

## 11. Design Decisions and Trade-offs

### Why SVM for mouth detection instead of a threshold on MAR?

A fixed MAR threshold would require manual tuning and would not generalize across different people (some people have naturally wider mouths). The SVM learns the boundary from actual data, adapting to the distribution of MAR values seen during training. The StandardScaler further normalizes for inter-person variability.

### Why three cycles to flag talking instead of one open event?

A single mouth opening could be a yawn, a cough, or a deep breath. Three rapid open-close cycles is a pattern specific to speech. This reduces false positives significantly at the cost of a short detection delay (the student must speak for at least a few syllables before flagging).

### Why AND logic for the absence counter?

Either detector alone has failure modes in different directions:
- Haar alone: misses tilted faces, and has false positives on background face-like patterns
- MediaPipe alone: occasional single-frame flickering re-detections reset the counter

Requiring both to agree for a counter reset eliminates both failure modes. The counter only resets when there is strong evidence the face is genuinely present.

### Why calibrate instead of using fixed thresholds?

Students sit at different distances and angles from the webcam. A 10-degree leftward yaw for one student might be their neutral straight-ahead position. Calibration captures each student's individual baseline and makes all detection relative to it.

### Why episode logging instead of per-frame logging?

Per-frame logs would produce thousands of rows per exam session and would be meaningless to a human reviewer. An instructor reviewing the log cares about "the student looked away for 8 seconds at 10:23 AM," not "frame 14,523 was flagged." Episode logging produces one interpretable row per behavioral event.

### Why a background thread for the camera loop?

`cv2.VideoCapture.read()` is a blocking call — it waits until the camera delivers a frame. Calling it on the Tkinter main thread would freeze the entire UI (buttons, labels, all redraws stop) while waiting. The background thread keeps the camera running continuously while Tkinter's mainloop handles UI events normally. Communication between the thread and the UI uses `self.after(0, ...)`, which is the thread-safe way to schedule code on the Tkinter main thread.
