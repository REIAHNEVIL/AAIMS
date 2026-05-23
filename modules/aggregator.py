# modules/aggregator.py
# The core processing module for AAIMS.
# Runs every webcam frame through all three detection modules
# and produces a unified result for the GUI to display.

import cv2
import mediapipe as mp
import csv
import datetime
import os

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
        Returns a dictionary of all detection results for the GUI.
        """
        img_h, img_w = frame.shape[:2]

        # convert to grayscale for face absence detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # convert to RGB for mediapipe
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # default values if no face is detected
        pitch        = 0.0
        yaw          = 0.0
        roll         = 0.0
        mar          = 0.0
        face_present = 0
        head_flag    = False
        mouth_flag   = False
        direction    = 'No Face'

        # run mediapipe face mesh
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_present = 1
            landmarks    = results.multi_face_landmarks[0]

            # head pose estimation
            angles = self.head_pose.get_angles(
                landmarks, img_w, img_h)

            if angles[0] is not None:
                pitch, yaw, roll = angles[0], angles[1], angles[2]
                direction = self.head_pose.get_direction(pitch, yaw)
                head_flag = self.head_pose.check_flag(pitch, yaw)

            # mouth movement detection
            mouth_flag = self.mouth.check_flag(
                landmarks.landmark, img_w, img_h)
            mar = self.mouth.last_mar

        # face absence detection runs every frame regardless
        absence_flag = self.face_absence.check_flag(gray)

        # determine flagged event types
        flagged_events = []
        if head_flag:
            flagged_events.append('HEAD_TURN')
        if absence_flag:
            flagged_events.append('FACE_ABSENT')
        if mouth_flag:
            flagged_events.append('MOUTH_OPEN')

        # log and alert if any flag is active
        if flagged_events:
            for event in flagged_events:
                self.log_event(event)
            if self.alert_callback:
                self.alert_callback(', '.join(flagged_events))

        return {
            'pitch':        pitch,
            'yaw':          yaw,
            'roll':         roll,
            'mar':          mar,
            'face_present': face_present,
            'direction':    direction,
            'head_flag':    head_flag,
            'absence_flag': absence_flag,
            'mouth_flag':   mouth_flag,
            'flagged':      len(flagged_events) > 0
        }