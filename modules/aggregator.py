# modules/aggregator.py
# The core processing module for AAIMS.
# Runs every webcam frame through all three detection modules
# and produces a unified result for the GUI to display.

import cv2
import mediapipe as mp
import csv
import datetime
import os
import time

from modules.head_pose import HeadPoseDetector
from modules.face_absence import FaceAbsenceDetector
from modules.mouth_detection import MouthMovementDetector

class AaimsProcessor:

    def __init__(self):
        # initialize all three detection modules
        self.head_pose    = HeadPoseDetector()
        self.face_absence = FaceAbsenceDetector()
        self.mouth        = MouthMovementDetector()

        # mediapipe face mesh for landmark extraction
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # callback function set by the GUI to show alerts
        self.alert_callback = None

        # tracks when the last mouth flag occurred for display
        self.mouth_alert_time    = 0
        self.MOUTH_ALERT_DISPLAY = 3  # seconds to keep alert visible

        # set up the integrity log file for this session
        os.makedirs('logs', exist_ok=True)
        self.log_path = f'logs/session_{datetime.date.today()}.csv'
        self._init_log()

        print('AaimsProcessor initialized successfully.')

    def _init_log(self):
        """Creates the log file with headers."""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'event_type'])

    def log_event(self, event_type):
        """Writes a flagged event with timestamp to the log file."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type])

    def process_frame(self, frame):
        """
        Processes one webcam frame through the full AAIMS pipeline.
        Haar Cascade is the single source of truth for face presence.
        MediaPipe only runs when Haar Cascade confirms a face is present.
        """
        img_h, img_w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # default values
        pitch        = 0.0
        yaw          = 0.0
        roll         = 0.0
        mar          = 0.0
        face_present = 0
        head_flag    = False
        mouth_flag   = False
        direction    = 'No Face'

        # step 1 — haar cascade decides if face is present
        # this is the single source of truth for face presence
        haar_says_present = self.face_absence.predict(gray)

        if haar_says_present == 1:
            face_present = 1
            self.face_absence.absence_counter = 0

            # step 2 — mediapipe only runs when haar confirms face
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                # head pose estimation
                angles = self.head_pose.get_angles(
                    landmarks, img_w, img_h)

                if angles[0] is not None:
                    pitch     = angles[0]
                    yaw       = angles[1]
                    roll      = angles[2]
                    direction = self.head_pose.get_direction(pitch, yaw)
                    head_flag = self.head_pose.check_flag(pitch, yaw)

                # mouth movement detection
                mouth_flag = self.mouth.check_flag(
                    landmarks.landmark, img_w, img_h)
                mar = self.mouth.last_mar

        else:
            # haar says no face — increment absence counter
            face_present = 0
            direction    = 'No Face'
            self.face_absence.absence_counter += 1

            # reset mouth values since no face is visible
            self.mouth.last_mar        = 0.0
            self.mouth.last_prediction = 0
            self.mouth.flag_counter    = 0
            self.mouth.cycle_counter   = 0
            mar = 0.0

        # step 3 — check absence flag
        absence_flag = (
            self.face_absence.absence_counter >=
            self.face_absence.ABSENCE_THRESHOLD
        )

        # step 4 — start mouth alert timer when mouth is flagged
        if mouth_flag:
            self.mouth_alert_time = time.time()

        # mouth alert stays visible for 5 seconds after flagging
        mouth_alert_active = (
            time.time() - self.mouth_alert_time
        ) < self.MOUTH_ALERT_DISPLAY

        # step 5 — collect all flagged events
        flagged_events = []
        if head_flag:    flagged_events.append('HEAD_TURN')
        if absence_flag: flagged_events.append('FACE_ABSENT')
        if mouth_flag:   flagged_events.append('MOUTH_OPEN')

        # step 6 — log and alert
        if flagged_events:
            for event in flagged_events:
                self.log_event(event)
            if self.alert_callback:
                self.alert_callback(', '.join(flagged_events))

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