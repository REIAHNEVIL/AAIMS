# modules/aggregator.py
# The core processing module for AAIMS.
# Runs every webcam frame through all three detection modules
# and produces a unified result for the GUI to display.
# Logs behavioral events as episodes with start time,
# end time, and duration instead of per-frame entries.

import cv2
import mediapipe as mp
import csv
import datetime
import os
import time

from modules.head_pose import HeadPoseDetector
from modules.face_absence import FaceAbsenceDetector
from modules.mouth_detection import MouthMovementDetector

# human readable event labels for the integrity log
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

        # tracks active episodes per event type
        # each entry stores the start time of an ongoing event
        self.active_episodes = {
            'HEAD_TURN':   None,
            'FACE_ABSENT': None,
            'MOUTH_OPEN':  None,
        }

        # mouth alert display timer
        self.mouth_alert_time    = 0
        self.MOUTH_ALERT_DISPLAY = 5

        # set up log file
        os.makedirs('logs', exist_ok=True)
        self.log_path = f'logs/session_{datetime.date.today()}.csv'
        self._init_log()

        print('AaimsProcessor initialized successfully.')

    def _init_log(self):
        """Creates the log file with column headers."""
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
        """Records the start of a new behavioral event episode."""
        if self.active_episodes[event_type] is None:
            self.active_episodes[event_type] = datetime.datetime.now()

    def _end_episode(self, event_type):
        """
        Records the end of a behavioral event episode.
        Writes one row to the log with start time, end time,
        duration, and a human readable description.
        """
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
        """
        Starts or ends event episodes based on current flags.
        An episode starts when a flag turns on and ends
        when that flag turns off.
        """
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
        img_h, img_w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pitch        = 0.0
        yaw          = 0.0
        roll         = 0.0
        mar          = 0.0
        face_present = 0
        head_flag    = False
        mouth_flag   = False
        direction    = 'No Face'

        # haar cascade is the source of truth for face presence
        haar_says_present = self.face_absence.predict(gray)

        if haar_says_present == 1:
            face_present = 1
            self.face_absence.absence_counter = 0

            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                angles = self.head_pose.get_angles(
                    landmarks, img_w, img_h)

                if angles[0] is not None:
                    pitch     = angles[0]
                    yaw       = angles[1]
                    roll      = angles[2]
                    direction = self.head_pose.get_direction(
                        pitch, yaw)
                    head_flag = self.head_pose.check_flag(
                        pitch, yaw)

                mouth_flag = self.mouth.check_flag(
                    landmarks.landmark, img_w, img_h)
                mar = self.mouth.last_mar

        else:
            face_present = 0
            direction    = 'No Face'
            self.face_absence.absence_counter += 1

            self.mouth.last_mar        = 0.0
            self.mouth.last_prediction = 0
            self.mouth.flag_counter    = 0
            self.mouth.cycle_counter   = 0
            mar = 0.0

        absence_flag = (
            self.face_absence.absence_counter >=
            self.face_absence.ABSENCE_THRESHOLD
        )

        # update episode tracking
        self._update_episodes(head_flag, absence_flag, mouth_flag)

        # mouth alert display timer
        if mouth_flag:
            self.mouth_alert_time = time.time()

        mouth_alert_active = (
            time.time() - self.mouth_alert_time
        ) < self.MOUTH_ALERT_DISPLAY

        # collect active flags for GUI alert
        flagged_events = []
        if head_flag:    flagged_events.append('HEAD_TURN')
        if absence_flag: flagged_events.append('FACE_ABSENT')
        if mouth_flag:   flagged_events.append('MOUTH_OPEN')

        if flagged_events and self.alert_callback:
            self.alert_callback(
                ', '.join(EVENT_LABELS[e]
                          for e in flagged_events))

        return {
            'pitch':              pitch,
            'yaw':                yaw,
            'roll':               roll,
            'mar':                mar,
            'face_present':       face_present,
            'direction':          direction,
            'head_flag':          head_flag,
            'absence_flag':       absence_flag,
            'mouth_flag':         mouth_flag,
            'mouth_alert_active': mouth_alert_active,
            'flagged':            len(flagged_events) > 0
        }