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
        self.mouth_classifier = joblib.load(model_path)
        self.feature_scaler   = joblib.load(scaler_path)

        # counts consecutive frames where mouth is classified as open
        self.open_frame_count         = 0
        self.MIN_OPEN_FRAMES_PER_CYCLE = 5   # frames needed to count as one open cycle

        # counts how many open-close cycles have occurred
        # talking produces multiple cycles, a yawn produces one
        self.open_close_cycle_count  = 0
        self.CYCLES_TO_FLAG_TALKING  = 3   # cycles needed to flag as talking

        # last raw MAR value for display purposes
        self.last_mar = 0.0

        # last SVM prediction for display purposes (0 = closed, 1 = open)
        self.current_mouth_state = 0

        print('MouthMovementDetector loaded successfully.')

    def compute_mar(self, landmarks, image_width, image_height):
        """
        Computes the Mouth Aspect Ratio (MAR) from lip landmarks.
        MAR = vertical lip distance / horizontal mouth width.
        Higher MAR means mouth is more open.
        """
        def landmark_to_pixel(landmark_index):
            landmark = landmarks[landmark_index]
            return np.array([landmark.x * image_width, landmark.y * image_height])

        vertical_lip_distance  = np.linalg.norm(
            landmark_to_pixel(UPPER_LIP) - landmark_to_pixel(LOWER_LIP))
        horizontal_mouth_width = np.linalg.norm(
            landmark_to_pixel(LEFT_CORNER) - landmark_to_pixel(RIGHT_CORNER))

        # small epsilon avoids division by zero on degenerate frames
        return vertical_lip_distance / (horizontal_mouth_width + 1e-6)

    def detect_talking(self, landmarks, image_width, image_height):
        """
        Detects repeated mouth opening patterns consistent
        with verbal communication.

        Counts open-close cycles. When CYCLES_TO_FLAG_TALKING cycles
        are reached, flags once then resets to detect the
        next episode of talking.

        Returns True only on the frame the threshold is crossed.
        """
        mouth_aspect_ratio = self.compute_mar(landmarks, image_width, image_height)
        self.last_mar = mouth_aspect_ratio

        normalized_mar           = self.feature_scaler.transform([[mouth_aspect_ratio]])
        self.current_mouth_state = self.mouth_classifier.predict(normalized_mar)[0]

        talking_detected = False

        if self.current_mouth_state == 1:
            self.open_frame_count += 1
        else:
            # mouth just closed — check if it sustained long enough to count as a cycle
            if self.open_frame_count >= self.MIN_OPEN_FRAMES_PER_CYCLE:
                self.open_close_cycle_count += 1

                if self.open_close_cycle_count >= self.CYCLES_TO_FLAG_TALKING:
                    talking_detected = True
                    # reset so the next talking episode can be detected independently
                    self.open_close_cycle_count = 0

            self.open_frame_count = 0

        return talking_detected

    def reset(self):
        """Resets all counters — call at session start."""
        self.open_frame_count       = 0
        self.open_close_cycle_count = 0
        self.last_mar               = 0.0
        self.current_mouth_state    = 0
