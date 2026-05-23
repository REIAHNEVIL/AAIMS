# modules/mouth_detection.py
# Mouth movement detection module for AAIMS.
# Computes the Mouth Aspect Ratio (MAR) from MediaPipe lip
# landmarks and uses the trained SVM to classify whether
# the mouth is open or closed.
# Flags sustained mouth-open events as potential verbal
# communication during the examination.

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

        # consecutive open-mouth frame counter
        self.flag_counter    = 0
        self.FRAME_THRESHOLD = 5  # ~0.5 seconds at 30fps

        # last raw MAR value for display purposes
        self.last_mar = 0.0

        # last prediction for display purposes
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

        vertical   = np.linalg.norm(get_point(UPPER_LIP) - get_point(LOWER_LIP))
        horizontal = np.linalg.norm(get_point(LEFT_CORNER) - get_point(RIGHT_CORNER))

        # add small value to avoid division by zero
        return vertical / (horizontal + 1e-6)

    def check_flag(self, landmarks, img_w, img_h):
        """
        Computes MAR from lip landmarks and classifies
        mouth state using the trained SVM classifier.
        Returns True if mouth has been open for more than
        FRAME_THRESHOLD consecutive frames.
        """
        mar = self.compute_mar(landmarks, img_w, img_h)
        self.last_mar = mar

        # scale MAR using the same scaler used during training
        mar_scaled = self.scaler.transform([[mar]])

        # predict: 0 = closed, 1 = open
        self.last_prediction = self.svm.predict(mar_scaled)[0]

        if self.last_prediction == 1:
            self.flag_counter += 1
        else:
            self.flag_counter = 0

        return self.flag_counter >= self.FRAME_THRESHOLD
    
    def reset(self):
        """Resets counters — call at session start."""
        self.flag_counter    = 0
        self.last_mar        = 0.0
        self.last_prediction = 0