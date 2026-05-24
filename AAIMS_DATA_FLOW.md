# AAIMS — Data Flow Walkthrough
## From Session Start to Integrity Log

This document traces exactly what data exists, what shape it takes, and how it transforms at every step of a full AAIMS session. Follow this top to bottom as if you are watching one complete exam session run.

---

## Stage 1 — Program Startup (`main.py`)

```python
processor = AaimsProcessor()
```

This one line triggers a chain of initializations before any GUI appears.

### Inside `AaimsProcessor.__init__()`

**Three detection objects are created:**

```python
self.head_pose    = HeadPoseDetector()
self.face_absence = FaceAbsenceDetector()
self.mouth        = MouthMovementDetector()
```

`HeadPoseDetector.__init__()` sets:
```
self.yaw_threshold   = 10
self.pitch_threshold = 10
self.pitch_baseline  = 0      ← will be replaced during calibration
self.yaw_baseline    = 0      ← will be replaced during calibration
self.sustained_off_screen_frames = 0
self.FRAMES_TO_FLAG_HEAD_TURN    = 20
```

`FaceAbsenceDetector.__init__()` loads the Haar Cascade XML file from disk into memory and sets:
```
self.absence_counter    = 0
self.ABSENCE_THRESHOLD  = 30
self.current_face_state = 1
```

`MouthMovementDetector.__init__()` reads two `.pkl` files from disk:
```
self.mouth_classifier ← deserialized SVM object  (from svm_mouth_classifier.pkl)
self.feature_scaler   ← deserialized Scaler object (from svm_mouth_scaler.pkl)
self.open_frame_count          = 0
self.MIN_OPEN_FRAMES_PER_CYCLE = 5
self.open_close_cycle_count    = 0
self.CYCLES_TO_FLAG_TALKING    = 3
self.last_mar            = 0.0
self.current_mouth_state = 0
```

**MediaPipe FaceMesh is initialized:**
```python
self.face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```
This loads a TFLite neural network model into memory. It accepts RGB frames and returns up to 1 set of 478 face landmarks.

**The log file is created:**
```python
self.log_path = f'logs/session_{datetime.date.today()}.csv'
```
Example: `logs/session_2026-05-24.csv`

`_init_log()` writes the header row immediately:
```
start_time, end_time, duration_seconds, event_type, description
```
The file exists and is empty (header only) before any detection runs.

**Episode tracking is initialized:**
```python
self.active_episodes = {
    'HEAD_TURN':   None,
    'FACE_ABSENT': None,
    'MOUTH_OPEN':  None,
}
```
`None` means that event type is not currently active.

---

## Stage 2 — Notice Screen

No data is processed or transformed here. The proctor reads the notice, student clicks Acknowledge, and the screen destroys itself. The `processor` object sits idle.

---

## Stage 3 — Session Setup (`gui/session_setup.py`)

The proctor fills in the form and clicks Start. The form data is collected into a plain Python dict:

```python
config = {
    'student_id':        'Juan dela Cruz',
    'exam_name':         'Midterm Examination',
    'workstation':       '3',
    'yaw_threshold':     10,
    'absence_threshold': 30
}
```

This dict is passed by reference into every subsequent screen. It is never mutated after this point — it is read-only metadata used for display labels and log export filenames.

---

## Stage 4 — Calibration (`gui/calibration_screen.py`)

This stage is the only place where the `processor`'s internal baseline values change.

### Webcam Opens

```python
self.cap = cv2.VideoCapture(0)
```

A separate MediaPipe FaceMesh instance is created just for calibration (the main `processor.face_mesh` is reserved for the live session):

```python
self.face_mesh = mp.solutions.face_mesh.FaceMesh(...)
```

### The Preview Loop (before Start is clicked)

Every 33ms, `_update_preview()` fires via `self.after(33, ...)`:

```
cap.read()
    → success (bool), frame (numpy array, shape: [H, W, 3], dtype: uint8, BGR)

cv2.cvtColor(frame, BGR→RGB)
    → rgb_frame (numpy array, shape: [H, W, 3], dtype: uint8, RGB)

cv2.resize(rgb_frame, (480, 300))
    → resized_frame (numpy array, shape: [300, 480, 3])

ImageTk.PhotoImage(Image.fromarray(resized_frame))
    → display_image (Tkinter-compatible image object)

video_label.configure(image=display_image)
    → frame appears on screen
```

### Calibration Frames (after Start is clicked)

For each of the 100 frames, `_process_calibration_frame(frame)` runs:

```
frame (BGR numpy array)
    │
    ├── cv2.cvtColor(BGR→RGB) → rgb_frame
    │
    └── face_mesh.process(rgb_frame)
            → results.multi_face_landmarks
                    │
                    └── landmarks = results.multi_face_landmarks[0]
                            │
                            ├── head_pose.get_angles(landmarks, w, h)
                            │       → (pitch, yaw, roll, nose_2d, nose_3d, rot_vec, trans_vec)
                            │       → angle_buffer.append( (pitch, yaw) )
                            │
                            └── mouth.compute_mar(landmarks.landmark, w, h)
                                    → mouth_aspect_ratio (float, e.g. 0.042)
                                    → mar_buffer.append(0.042)
```

After 100 frames:

```
angle_buffer = [(p0,y0), (p1,y1), ..., (p99,y99)]   ← 100 (pitch, yaw) pairs
mar_buffer   = [0.038, 0.041, 0.039, ...]            ← 100 MAR floats
```

`_finish()` distills these into scalar baselines:

```python
head_pose.calibrate(angle_buffer)
    → self.pitch_baseline = np.mean([a[0] for a in angle_buffer])   e.g.  2.31
    → self.yaw_baseline   = np.mean([a[1] for a in angle_buffer])   e.g. -1.05

mouth.mar_baseline = float(np.mean(mar_buffer))                      e.g.  0.040
```

After this point `pitch_baseline`, `yaw_baseline`, and `mar_baseline` are fixed for the entire session. The calibration FaceMesh and webcam are released.

---

## Stage 5 — Live Monitoring (`gui/proctor_dashboard.py`)

This is the main loop. A background thread calls `process_frame()` continuously.

### The Camera Thread

```python
def _camera_loop(self):
    while self.running:
        success, frame = self.cap.read()
        result = self.processor.process_frame(frame)
        self.after(0, self._update_gui, frame, result)
```

`frame` is a raw BGR numpy array from the webcam, shape approximately `(480, 640, 3)`.

---

### Inside `process_frame(frame)` — One Frame, Full Trace

We follow a single frame through the entire function.

#### Step 1 — Dimension Extraction

```python
image_height, image_width = frame.shape[:2]
# e.g. image_height=480, image_width=640
```

#### Step 2 — Color Conversions

```python
grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# shape: (480, 640)  — 2D, single channel intensity values 0–255
# used by: Haar Cascade

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# shape: (480, 640, 3)  — same as input but R and B channels swapped
# used by: MediaPipe (which expects RGB, not BGR)
```

#### Step 3 — MediaPipe Face Mesh

```python
results = self.face_mesh.process(rgb_frame)
mediapipe_detects_face = bool(results.multi_face_landmarks)
```

If a face is found, `results.multi_face_landmarks` is a list with one entry (we set `max_num_faces=1`). Each entry contains 478 landmark objects. Each landmark has:

```
landmark.x   ← normalized horizontal position (0.0 = left edge, 1.0 = right edge)
landmark.y   ← normalized vertical position (0.0 = top, 1.0 = bottom)
landmark.z   ← relative depth (negative = closer to camera)
```

To get pixel coordinates:
```python
x_pixel = landmark.x * image_width   # e.g. 0.423 * 640 = 271
y_pixel = landmark.y * image_height  # e.g. 0.381 * 480 = 183
```

#### Step 4 — Haar Cascade

```python
haar_detects_face = self.face_absence.predict(grayscale_frame) == 1
```

Inside `predict()`:
```python
faces = self.face_cascade.detectMultiScale(
    grayscale_frame, scaleFactor=1.05, minNeighbors=4, minSize=(30,30)
)
# faces is either:
#   a numpy array of shape (N, 4) where each row is [x, y, w, h]
#   or an empty tuple () if nothing found
return 1 if len(faces) > 0 else 0
```

`haar_detects_face` is now a plain `bool`.

#### Step 5 — Absence Counter Update

```python
if haar_detects_face and mediapipe_detects_face:
    self.face_absence.absence_counter = 0
else:
    self.face_absence.absence_counter += 1
```

Both must agree the face is present to reset. If either misses, the counter climbs. After 30 consecutive non-confirmed frames (~1 second), `absence_flag` will be True.

#### Step 6 — Head Pose (runs only if MediaPipe found landmarks)

```python
landmarks = results.multi_face_landmarks[0]

pitch, yaw, roll, *_ = self.head_pose.get_angles(
    landmarks, image_width, image_height)
```

Inside `get_angles()`:

**6a — Extract 6 key landmarks:**
```
landmark index 1   → nose tip
landmark index 33  → left eye outer corner
landmark index 61  → left mouth corner
landmark index 199 → chin
landmark index 263 → right eye outer corner
landmark index 291 → right mouth corner
```

For each, pixel coordinates are computed and stored in two lists:
```
face_2d = [[271,183], [198,162], [229,247], [265,318], [318,165], [302,250]]
           (6 rows, 2 columns — x,y pixel coordinates)

face_3d = [[271,183, -0.023], [198,162, -0.031], ...]
           (6 rows, 3 columns — x,y pixel + z depth from MediaPipe)
```

Both are converted to `np.float64` arrays.

**6b — Build the camera intrinsic matrix:**
```
focal_length = 1 * 640 = 640

camera_matrix = [[640,   0, 240],
                 [  0, 640, 320],
                 [  0,   0,   1]]
```

Principal point is assumed to be the image center (240, 320) for a 480×640 frame.

**6c — Run solvePnP:**
```python
solved, rotation_vector, translation_vector = cv2.solvePnP(
    face_3d, face_2d, camera_matrix, distortion_coefficients)
```
```
rotation_vector = [[-0.312], [0.085], [0.024]]   shape: (3,1)
   (axis-angle representation of the face's rotation in camera space)

translation_vector = [[23.4], [-18.1], [812.3]]  shape: (3,1)
   (position of face relative to camera, in the same unit as face_3d z)
```

**6d — Convert rotation vector → rotation matrix:**
```python
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
```
```
rotation_matrix = [[ 0.996,  0.023, -0.083],
                   [-0.019,  0.998,  0.055],
                   [ 0.085, -0.054,  0.995]]   shape: (3,3)
```

**6e — Extract Euler angles:**
```python
euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

pitch = euler_angles[0] * 360   →  e.g.  1.8  degrees
yaw   = euler_angles[1] * 360   →  e.g. -0.7  degrees
roll  = euler_angles[2] * 360   →  e.g.  0.3  degrees
```

**6f — Direction label:**
```python
direction = self.head_pose.get_direction(pitch, yaw)
```
Compares pitch and yaw against the 10-degree threshold → returns `"Forward"` (since both are within ±10°).

**6g — Sustained frame counter:**
```python
head_flag = self.head_pose.detect_head_turned(pitch, yaw)
```
```
pitch_deviation = 1.8 - 2.31 = -0.51   (within threshold)
yaw_deviation   = |-0.7 - (-1.05)| = 0.35   (within threshold)

horizontal_turn = False
looking_down    = False

→ sustained_off_screen_frames = 0   (reset)
→ head_flag = False   (0 < 20)
```

#### Step 7 — Mouth Detection

```python
mouth_flag = self.mouth.detect_talking(
    landmarks.landmark, image_width, image_height)
mouth_aspect_ratio = self.mouth.last_mar
```

Inside `detect_talking()`:

**7a — Compute MAR:**
```python
mouth_aspect_ratio = self.compute_mar(landmarks, image_width, image_height)
```

Four landmark pixel positions are extracted (indices 13, 14, 61, 291):
```
upper_lip_px   = [312, 278]
lower_lip_px   = [312, 295]
left_corner_px = [229, 284]
right_corner_px = [394, 284]

vertical_lip_distance  = ||[312,278] - [312,295]|| = 17.0 px
horizontal_mouth_width = ||[229,284] - [394,284]|| = 165.0 px

MAR = 17.0 / (165.0 + 1e-6) = 0.103
```

**7b — Scale the MAR:**
```python
normalized_mar = self.feature_scaler.transform([[0.103]])
# StandardScaler applies: (x - mean) / std
# e.g. mean=0.089, std=0.031 → (0.103 - 0.089) / 0.031 = 0.452
# normalized_mar = [[0.452]]
```

**7c — SVM prediction:**
```python
self.current_mouth_state = self.mouth_classifier.predict([[0.452]])[0]
# → 0  (closed)
```

**7d — Cycle counting (mouth closed on this frame):**
```python
# current_mouth_state == 0, so we enter the else branch
if self.open_frame_count >= self.MIN_OPEN_FRAMES_PER_CYCLE:  # 0 >= 5? No
    ...   # cycle not counted
self.open_frame_count = 0

talking_detected = False
```

No cycle is counted. `mouth_flag = False`.

#### Step 8 — Absence Flag

```python
absence_flag = (
    self.face_absence.absence_counter >= self.face_absence.ABSENCE_THRESHOLD
)
# e.g. 0 >= 30 → False
```

#### Step 9 — Episode Update

```python
self._update_episodes(head_flag=False, absence_flag=False, mouth_flag=False)
```

All flags are False, so `_update_episodes` checks each:
- `HEAD_TURN`: flag is False, `active_episodes['HEAD_TURN']` is None → do nothing
- `FACE_ABSENT`: same
- `MOUTH_OPEN`: same

No CSV rows are written.

#### Step 10 — Mouth Alert Timer

```python
# mouth_flag is False, so mouth_alert_time is not updated
mouth_alert_active = (time.time() - 0) < 5
# → True if less than 5 seconds since program start, else False
# In practice: False after the first 5 seconds of the session
```

#### Step 11 — Result Dict Assembly

```python
return {
    'pitch':              1.8,
    'yaw':               -0.7,
    'roll':               0.3,
    'mar':                0.103,
    'face_present':       1,
    'direction':          'Forward',
    'head_flag':          False,
    'absence_flag':       False,
    'mouth_flag':         False,
    'mouth_alert_active': False,
    'flagged':            False
}
```

This dict is passed to `_update_gui()` on the main thread.

---

### Back in `_update_gui(frame, result)`

```python
rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized_frame = cv2.resize(rgb_frame, (480, 360))
display_image = ImageTk.PhotoImage(Image.fromarray(resized_frame))
self.video_label.configure(image=display_image)
self.video_label.image = display_image
```

The raw frame (BGR) is converted and resized for display. `video_label.image = display_image` keeps a reference to prevent Python's garbage collector from deleting the image before Tkinter draws it.

**Status dots update:**
```
'face_present'      → result = 1 → green ●
'head_flag'         → result = False → green ●
'absence_flag'      → result = False → green ●
'mouth_alert_active'→ result = False → green ●
```

**Metric labels update:**
```
Direction:  Forward
Yaw:        -0.7 deg
Pitch:       1.8 deg
MAR:         0.1030
```

---

## Stage 5b — A Flagged Frame

Now let's trace a frame where the student has been looking left for 22 consecutive frames.

At this point: `sustained_off_screen_frames = 21` from the previous frame.

#### Inside `process_frame()`

`get_angles()` returns `pitch=3.1, yaw=18.4` (well above the 10-degree yaw threshold).

```python
direction = get_direction(3.1, 18.4)
# yaw=18.4 > yaw_threshold=10 → "Looking Left"

head_flag = detect_head_turned(3.1, 18.4)
# yaw_deviation = |18.4 - (-1.05)| = 19.45 > 10 → horizontal_turn = True
# sustained_off_screen_frames = 21 + 1 = 22
# 22 >= 20 → head_flag = True
```

#### `_update_episodes(head_flag=True, absence_flag=False, mouth_flag=False)`

```python
flags = {'HEAD_TURN': True, 'FACE_ABSENT': False, 'MOUTH_OPEN': False}

for event_type, is_active in flags.items():
    if is_active:
        self._start_episode(event_type)
```

For `HEAD_TURN`:
```python
def _start_episode('HEAD_TURN'):
    if self.active_episodes['HEAD_TURN'] is None:   # True — first flagged frame
        self.active_episodes['HEAD_TURN'] = datetime.datetime.now()
        # e.g. 2026-05-24 10:23:41.882
```

For `FACE_ABSENT` and `MOUTH_OPEN`:
```python
# is_active = False, active_episodes[type] = None → do nothing
```

No CSV row is written yet — the episode is open (in progress).

#### Result dict now contains:
```python
{
    'head_flag': True,
    'direction': 'Looking Left',
    'flagged':   True,
    ...
}
```

Status dot for `head_flag` → **red ●**
Overall color on the dashboard → **red**

---

## Stage 5c — The Student Looks Forward Again (Episode Ends)

Three frames later, `sustained_off_screen_frames` drops to 0.

```python
head_flag = detect_head_turned(2.1, 3.2)
# yaw_deviation = |3.2 - (-1.05)| = 4.25 < 10 → horizontal_turn = False
# looking_down = False
# sustained_off_screen_frames = 0   (reset)
# 0 >= 20 → head_flag = False
```

#### `_update_episodes(head_flag=False, ...)`

```python
# is_active = False, active_episodes['HEAD_TURN'] = datetime(10:23:41.882)
# → active_episodes is not None, so call _end_episode
```

Inside `_end_episode('HEAD_TURN')`:
```python
start    = 2026-05-24 10:23:41.882
end      = 2026-05-24 10:23:42.784    (about 0.9 seconds later)
duration = (end - start).seconds = 0
```

A row is appended to the CSV:
```
2026-05-24 10:23:41, 2026-05-24 10:23:42, 0, HEAD_TURN, Student turned head away from screen
```

`active_episodes['HEAD_TURN']` is reset to `None`.

---

## Stage 5d — Mouth Movement Detection (Multi-Frame Trace)

Let us trace several frames of the student talking.

**Frame A** — mouth opens:
```
MAR = 0.19 → scaled = 3.23 → SVM → 1 (open)
current_mouth_state = 1
open_frame_count += 1 → open_frame_count = 1
talking_detected = False
```

**Frames B–F** — mouth stays open (5 more frames):
```
open_frame_count = 2, 3, 4, 5, 6
```

**Frame G** — mouth closes:
```
MAR = 0.06 → scaled = -0.94 → SVM → 0 (closed)
current_mouth_state = 0

# else branch:
open_frame_count (6) >= MIN_OPEN_FRAMES_PER_CYCLE (5) → True
open_close_cycle_count += 1 → open_close_cycle_count = 1

open_close_cycle_count (1) >= CYCLES_TO_FLAG_TALKING (3) → False
open_frame_count = 0

talking_detected = False
```

This repeats. After **cycle 2** ends:
```
open_close_cycle_count = 2
talking_detected = False
```

After **cycle 3** ends:
```
open_close_cycle_count (3) >= CYCLES_TO_FLAG_TALKING (3) → True
talking_detected = True
open_close_cycle_count = 0   ← reset for next episode detection
open_frame_count = 0
```

`mouth_flag = True` for exactly **one frame**.

Back in `process_frame()`:
```python
if mouth_flag:
    self.mouth_alert_time = time.time()
```

`mouth_alert_active` will now be True for the next 5 seconds even though `mouth_flag` only fired once.

`_start_episode('MOUTH_OPEN')` records the timestamp. The episode closes on the next frame (since `mouth_flag` is False again), writing one CSV row immediately.

---

## Stage 6 — Session End

The proctor clicks "End Session."

```python
def _end_session(self):
    self.running = False    # signals camera thread to stop
    self.cap.release()      # releases the webcam
    self.destroy()          # destroys the Tkinter window
```

Any active episodes in `active_episodes` that are still open (not None) will **not** be automatically closed — they remain partial. Only episodes that have ended (flag turned off) produce CSV rows. This is a known limitation: if the student is still looking away when End Session is clicked, that episode is not logged.

---

## Stage 7 — Integrity Log (`gui/integrity_log_screen.py`)

```python
session_log = pd.read_csv(self.processor.log_path)
```

`pandas` reads the CSV into a DataFrame. Each row is one episode. Example DataFrame:

| start_time | end_time | duration_seconds | event_type | description |
|---|---|---|---|---|
| 2026-05-24 10:23:41 | 2026-05-24 10:23:42 | 0 | HEAD_TURN | Student turned head away from screen |
| 2026-05-24 10:31:05 | 2026-05-24 10:31:09 | 4 | HEAD_TURN | Student turned head away from screen |
| 2026-05-24 10:45:22 | 2026-05-24 10:45:23 | 1 | MOUTH_OPEN | Repeated mouth movement detected |

```python
total_episodes       = len(session_log)                                    # 3
head_turn_count      = len(session_log[session_log['event_type'] == 'HEAD_TURN'])    # 2
face_absent_count    = len(session_log[session_log['event_type'] == 'FACE_ABSENT'])  # 0
mouth_movement_count = len(session_log[session_log['event_type'] == 'MOUTH_OPEN'])   # 1
```

These counts populate the summary label.

Each row is inserted into the Treeview:
```python
for _, episode in session_log.iterrows():
    self.tree.insert('', 'end',
        values=(
            episode['start_time'],
            episode['end_time'],
            f"{episode['duration_seconds']}s",
            episode['description']
        ),
        tags=('flagged',))
```

The `tags=('flagged',)` applies the pink row background configured earlier:
```python
self.tree.tag_configure('flagged', background='#FFE0E0')
```

---

## Stage 8 — Export

```python
destination_path = os.path.join(desktop_directory, filename)
shutil.copy(self.processor.log_path, destination_path)
```

`shutil.copy` makes a byte-for-byte copy of the CSV from `logs/session_2026-05-24.csv` to the Desktop with a name like `AAIMS_Log_Juan_dela_Cruz_Midterm_Examination.csv`. The original log file is not moved or deleted.

---

## Complete Data Shape Summary

| Stage | Data | Type | Shape / Example |
|---|---|---|---|
| Webcam raw frame | BGR image | `np.ndarray` | `(480, 640, 3)` uint8 |
| Grayscale frame | intensity image | `np.ndarray` | `(480, 640)` uint8 |
| RGB frame | RGB image | `np.ndarray` | `(480, 640, 3)` uint8 |
| MediaPipe landmarks | 478 landmark objects | list | `landmark.x=0.423, .y=0.381, .z=-0.023` |
| face_2d | pixel coords of 6 points | `np.ndarray` | `(6, 2)` float64 |
| face_3d | pixel+depth of 6 points | `np.ndarray` | `(6, 3)` float64 |
| camera_matrix | pinhole intrinsics | `np.ndarray` | `(3, 3)` float64 |
| rotation_vector | axis-angle rotation | `np.ndarray` | `(3, 1)` float64 |
| rotation_matrix | 3D rotation matrix | `np.ndarray` | `(3, 3)` float64 |
| euler_angles (raw) | RQ decomposition output | `np.ndarray` | `(3,)` float64, tiny values |
| pitch / yaw / roll | degrees | `float` | `1.8`, `-0.7`, `0.3` |
| MAR (raw) | mouth openness ratio | `float` | `0.103` |
| MAR (scaled) | standardized MAR | `np.ndarray` | `[[0.452]]` shape `(1,1)` |
| SVM prediction | open/closed | `int` | `0` or `1` |
| Haar detection | face found | `int` | `0` or `1` |
| absence_counter | frames without confirmed face | `int` | `0` to `∞` |
| head_flag | head turned too long | `bool` | `True` / `False` |
| absence_flag | face absent too long | `bool` | `True` / `False` |
| mouth_flag | talking detected | `bool` | `True` (one frame only) |
| result dict | all outputs per frame | `dict` | 11 keys |
| active_episodes entry | episode start time | `datetime` or `None` | `2026-05-24 10:23:41.882` |
| CSV row | one logged episode | list of 5 values | strings and int |
| session_log | full log as table | `pd.DataFrame` | N rows × 5 columns |

---

## Counter Behavior Quick Reference

### `sustained_off_screen_frames` (head pose)
```
Increments: every frame where yaw or pitch deviation exceeds threshold
Resets to 0: every frame where head is within threshold
Triggers:   when value reaches FRAMES_TO_FLAG_HEAD_TURN (20)
Effect:     head_flag = True while value >= 20
```

### `absence_counter` (face absence)
```
Increments: every frame where Haar AND MediaPipe do not both detect face
Resets to 0: every frame where Haar AND MediaPipe both detect face
Triggers:   when value reaches ABSENCE_THRESHOLD (30)
Effect:     absence_flag = True (stays True until counter resets)
```

### `open_frame_count` (mouth, counts frames mouth is open)
```
Increments: every frame where SVM predicts open (1)
Resets to 0: when mouth closes (SVM predicts 0), regardless of count
Triggers:   when mouth closes AND count >= MIN_OPEN_FRAMES_PER_CYCLE (5)
             → increments open_close_cycle_count
```

### `open_close_cycle_count` (mouth, counts full open-close cycles)
```
Increments: each time a qualifying open-close cycle completes
Resets to 0: when it reaches CYCLES_TO_FLAG_TALKING (3) and fires
             → mouth_flag = True for exactly that one frame
Does NOT reset: on a failed cycle (open_frame_count < 5)
```
