import numpy as np
import joblib

UPPER_LIP    = 13
LOWER_LIP    = 14
LEFT_CORNER  = 61
RIGHT_CORNER = 291

class MouthMovementDetector:

    def __init__(
        self,
        model_path ='models/svm_mouth_classifier.pkl',
        scaler_path='models/svm_mouth_scaler.pkl'
    ):
        self.mouth_classifier = joblib.load(model_path)
        self.feature_scaler   = joblib.load(scaler_path)

        self.open_frame_count          = 0
        self.MIN_OPEN_FRAMES_PER_CYCLE = 5

        self.open_close_cycle_count = 0
        self.CYCLES_TO_FLAG_TALKING = 3

        self.last_mar            = 0.0
        self.current_mouth_state = 0

        print('MouthMovementDetector loaded successfully.')

    def compute_mar(self, landmarks, image_width, image_height):
        def landmark_to_pixel(landmark_index):
            landmark = landmarks[landmark_index]
            return np.array([landmark.x * image_width, landmark.y * image_height])

        vertical_lip_distance  = np.linalg.norm(
            landmark_to_pixel(UPPER_LIP) - landmark_to_pixel(LOWER_LIP))
        horizontal_mouth_width = np.linalg.norm(
            landmark_to_pixel(LEFT_CORNER) - landmark_to_pixel(RIGHT_CORNER))

        return vertical_lip_distance / (horizontal_mouth_width + 1e-6)

    def detect_talking(self, landmarks, image_width, image_height):
        mouth_aspect_ratio = self.compute_mar(landmarks, image_width, image_height)
        self.last_mar      = mouth_aspect_ratio

        normalized_mar           = self.feature_scaler.transform([[mouth_aspect_ratio]])
        self.current_mouth_state = self.mouth_classifier.predict(normalized_mar)[0]

        talking_detected = False

        if self.current_mouth_state == 1:
            self.open_frame_count += 1
        else:
            if self.open_frame_count >= self.MIN_OPEN_FRAMES_PER_CYCLE:
                self.open_close_cycle_count += 1

                if self.open_close_cycle_count >= self.CYCLES_TO_FLAG_TALKING:
                    talking_detected            = True
                    self.open_close_cycle_count = 0

            self.open_frame_count = 0

        return talking_detected

    def reset(self):
        self.open_frame_count       = 0
        self.open_close_cycle_count = 0
        self.last_mar               = 0.0
        self.current_mouth_state    = 0
