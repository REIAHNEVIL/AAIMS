# modules/face_absence.py
# Face absence detection module for AAIMS.
# Uses OpenCV's Haar Cascade face detector to check
# whether a face is present in the current webcam frame.
# The fine-tuned CNN face classifier serves as the
# preprocessing gate, while the Haar Cascade handles
# mid-exam absence detection for better reliability
# across varying head angles and positions.

import cv2

class FaceAbsenceDetector:

    def __init__(self):
        # load OpenCV's built-in Haar Cascade face detector
        # this file comes bundled with OpenCV — no download needed
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # consecutive absence counter
        self.absence_counter   = 0
        self.ABSENCE_THRESHOLD = 30  # ~1 second at 30fps
        self.last_prediction   = 1  # add this line
        print('FaceAbsenceDetector loaded successfully.')

    def predict(self, gray_frame):
        """
        Detects whether a face is present in the grayscale frame
        using Haar Cascade.
        Returns 1 if face detected, 0 if not.
        """
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return 1 if len(faces) > 0 else 0

    def check_flag(self, gray_frame):
        """
        Checks the current frame for face presence and
        updates the absence counter accordingly.
        Returns True if face has been absent for more than
        ABSENCE_THRESHOLD consecutive frames.
        """
        self.last_prediction = self.predict(gray_frame)

        if self.last_prediction == 0:
            self.absence_counter += 1
        else:
            self.absence_counter = 0

        return self.absence_counter >= self.ABSENCE_THRESHOLD

    def reset(self):
        """Resets counters — call at session start."""
        self.absence_counter = 0