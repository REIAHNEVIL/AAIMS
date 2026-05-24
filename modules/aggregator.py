import cv2
import mediapipe as mp
import csv
import datetime
import os
import time

from modules.head_pose      import HeadPoseDetector
from modules.face_absence   import FaceAbsenceDetector
from modules.mouth_detection import MouthMovementDetector

EVENT_LABELS = {
    'HEAD_TURN':   'Student turned head away from screen',
    'FACE_ABSENT': 'Student face not visible in camera',
    'MOUTH_OPEN':  'Repeated mouth movement detected',
}

class AaimsProcessor:

    def __init__(self):
        self.head_pose    = HeadPoseDetector()
        self.face_absence = FaceAbsenceDetector()
        self.mouth        = MouthMovementDetector()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.alert_callback = None

        self.active_episodes = {
            'HEAD_TURN':   None,
            'FACE_ABSENT': None,
            'MOUTH_OPEN':  None,
        }

        self.mouth_alert_time    = 0
        self.MOUTH_ALERT_DISPLAY = 5

        os.makedirs('logs', exist_ok=True)
        self.log_path = f'logs/session_{datetime.date.today()}.csv'
        self._init_log()

        print('AaimsProcessor initialized successfully.')

    def _init_log(self):
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'start_time',
                'end_time',
                'duration_seconds',
                'event_type',
                'description'
            ])

    def _start_episode(self, event_type):
        if self.active_episodes[event_type] is None:
            self.active_episodes[event_type] = datetime.datetime.now()

    def _end_episode(self, event_type):
        start = self.active_episodes[event_type]
        if start is None:
            return

        end      = datetime.datetime.now()
        duration = (end - start).seconds

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                start.strftime('%Y-%m-%d %H:%M:%S'),
                end.strftime('%Y-%m-%d %H:%M:%S'),
                duration,
                event_type,
                EVENT_LABELS.get(event_type, event_type)
            ])

        self.active_episodes[event_type] = None

    def _update_episodes(self, head_flag, absence_flag, mouth_flag):
        flags = {
            'HEAD_TURN':   head_flag,
            'FACE_ABSENT': absence_flag,
            'MOUTH_OPEN':  mouth_flag,
        }

        for event_type, is_active in flags.items():
            if is_active:
                self._start_episode(event_type)
            else:
                if self.active_episodes[event_type] is not None:
                    self._end_episode(event_type)

    def process_frame(self, frame):
        image_height, image_width = frame.shape[:2]

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pitch              = 0.0
        yaw                = 0.0
        roll               = 0.0
        mouth_aspect_ratio = 0.0
        face_present       = 0
        head_flag          = False
        mouth_flag         = False
        direction          = 'No Face'

        results                = self.face_mesh.process(rgb_frame)
        mediapipe_detects_face = bool(results.multi_face_landmarks)
        haar_detects_face      = self.face_absence.predict(grayscale_frame) == 1

        if haar_detects_face and mediapipe_detects_face:
            self.face_absence.absence_counter = 0
        else:
            self.face_absence.absence_counter += 1

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

        else:
            face_present = 0
            direction    = 'No Face'

            self.mouth.last_mar               = 0.0
            self.mouth.current_mouth_state    = 0
            self.mouth.open_frame_count       = 0
            self.mouth.open_close_cycle_count = 0
            mouth_aspect_ratio = 0.0

        absence_flag = (
            self.face_absence.absence_counter >=
            self.face_absence.ABSENCE_THRESHOLD
        )

        self._update_episodes(head_flag, absence_flag, mouth_flag)

        if mouth_flag:
            self.mouth_alert_time = time.time()

        mouth_alert_active = (
            time.time() - self.mouth_alert_time
        ) < self.MOUTH_ALERT_DISPLAY

        flagged_events = []
        if head_flag:    flagged_events.append('HEAD_TURN')
        if absence_flag: flagged_events.append('FACE_ABSENT')
        if mouth_flag:   flagged_events.append('MOUTH_OPEN')

        if flagged_events and self.alert_callback:
            self.alert_callback(
                ', '.join(EVENT_LABELS[e] for e in flagged_events))

        return {
            'pitch':              pitch,
            'yaw':                yaw,
            'roll':               roll,
            'mar':                mouth_aspect_ratio,
            'face_present':       face_present,
            'direction':          direction,
            'head_flag':          head_flag,
            'absence_flag':       absence_flag,
            'mouth_flag':         mouth_flag,
            'mouth_alert_active': mouth_alert_active,
            'flagged':            len(flagged_events) > 0
        }
