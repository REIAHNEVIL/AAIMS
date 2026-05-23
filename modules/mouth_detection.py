# modules/mouth_detection.py
# Mouth movement detection module for AAIMS.
# Computes the Mouth Aspect Ratio (MAR) from MediaPipe lip
# landmarks and uses the trained SVM to classify whether
# the mouth is open or closed.
# Detects repeated open-close cycles consistent with talking
# rather than requiring one sustained open mouth.

import numpy as np
import joblib

# MediaPipe Face Mesh landmark indices for the lips
UPPER_LIP    = 13   # top center of upper lip
LOWER_LIP    = 14   # bottom center of lower lip
LEFT_CORNER  = 61   # left mouth corner
RIGHT_CORNER = 291  # right mouth corner

class MouthMovementDetector:

    def __init__(
        self,
        model_path ='models/svm_mouth_classifier.pkl',
        scaler_path='models/svm_mouth_scaler.pkl'
    ):
        # load the trained SVM and its scaler
        self.svm    = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # counts consecutive frames where mouth is open
        self.flag_counter    = 0
        self.FRAME_THRESHOLD = 5   # frames needed per open cycle

        # counts how many open-close cycles have occurred
        # talking produces multiple cycles, a yawn produces one
        self.cycle_counter   = 0
        self.CYCLE_THRESHOLD = 3   # cycles needed to flag

        # last raw MAR value for display purposes
        self.last_mar = 0.0

        # last SVM prediction for display purposes
        self.last_prediction = 0  # 0 = closed, 1 = open

        print('MouthMovementDetector loaded successfully.')

    def compute_mar(self, landmarks, img_w, img_h):
        """
        Computes the Mouth Aspect Ratio (MAR) from lip landmarks.
        MAR = vertical lip distance / horizontal mouth width.
        Higher MAR means mouth is more open.
        """
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        vertical   = np.linalg.norm(
            get_point(UPPER_LIP) - get_point(LOWER_LIP))
        horizontal = np.linalg.norm(
            get_point(LEFT_CORNER) - get_point(RIGHT_CORNER))

        # add small value to avoid division by zero
        return vertical / (horizontal + 1e-6)

    def check_flag(self, landmarks, img_w, img_h):
        """
        Detects repeated mouth opening patterns consistent
        with verbal communication.

        Counts open-close cycles. When CYCLE_THRESHOLD cycles
        are reached, flags once then resets to detect the
        next episode of talking.

        Returns True only on the frame the threshold is crossed.
        """
        mar = self.compute_mar(landmarks, img_w, img_h)
        self.last_mar = mar

        mar_scaled = self.scaler.transform([[mar]])
        self.last_prediction = self.svm.predict(mar_scaled)[0]

        flagged = False

        if self.last_prediction == 1:
            self.flag_counter += 1
        else:
            # mouth just closed — check if it counts as a cycle
            if self.flag_counter >= self.FRAME_THRESHOLD:
                self.cycle_counter += 1

                # check if enough cycles to flag
                if self.cycle_counter >= self.CYCLE_THRESHOLD:
                    flagged = True
                    # reset everything so next talking episode
                    # can be detected independently
                    self.cycle_counter = 0

            self.flag_counter = 0

        return flagged

    def reset(self):
        """Resets all counters — call at session start."""
        self.flag_counter    = 0
        self.cycle_counter   = 0
        self.last_mar        = 0.0
        self.last_prediction = 0