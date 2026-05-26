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
        self.feature_scaler = joblib.load(scaler_path)

        self.open_frame_count = 0
        self.MIN_OPEN_FRAMES_PER_CYCLE = 5

        self.open_close_cycle_count = 0
        self.CYCLES_TO_FLAG_TALKING = 3

        self.last_mar = 0.0
        self.current_mouth_state = 0
        self.previous_mouth_state = 0
        self.mar_baseline        = 0.0
        self.MAR_OPEN_MULTIPLIER = 1.5

        self.is_talking = False
        self.talking_inactivity_counter  = 0
        self.TALKING_INACTIVITY_THRESHOLD = 30

        print('MouthMovementDetector loaded successfully.')


    def compute_mar(self, landmarks, image_width, image_height):
        def landmark_to_pixel(landmark_index):
            landmark = landmarks[landmark_index]
            return np.array([landmark.x * image_width, landmark.y * image_height])

        vertical_lip_distance  = np.linalg.norm(landmark_to_pixel(UPPER_LIP) - landmark_to_pixel(LOWER_LIP))
        horizontal_mouth_width = np.linalg.norm(landmark_to_pixel(LEFT_CORNER) - landmark_to_pixel(RIGHT_CORNER))

        return vertical_lip_distance / (horizontal_mouth_width + 1e-6)


    def detect_talking(self, landmarks, image_width, image_height):
        mouth_aspect_ratio = self.compute_mar(landmarks, image_width, image_height)
        self.last_mar = mouth_aspect_ratio

        normalized_mar  = self.feature_scaler.transform([[mouth_aspect_ratio]])
        svm_prediction  = self.mouth_classifier.predict(normalized_mar)[0]

        baseline_open   = self.mar_baseline > 0 and mouth_aspect_ratio > self.mar_baseline * self.MAR_OPEN_MULTIPLIER
        self.current_mouth_state = 1 if (svm_prediction == 1 and (self.mar_baseline == 0.0 or baseline_open)) else 0

        cycle_completed = False

        if self.current_mouth_state == 1:
            self.open_frame_count += 1
            if self.previous_mouth_state == 0:
                print(f'[Mouth] Opened  | MAR: {mouth_aspect_ratio:.4f} | cycle: {self.open_close_cycle_count}/{self.CYCLES_TO_FLAG_TALKING}')
        else:
            if self.open_frame_count >= self.MIN_OPEN_FRAMES_PER_CYCLE:
                self.open_close_cycle_count += 1
                cycle_completed = True
                print(f'[Mouth] Cycle {self.open_close_cycle_count}/{self.CYCLES_TO_FLAG_TALKING} completed (open for {self.open_frame_count} frames)')

                if self.open_close_cycle_count >= self.CYCLES_TO_FLAG_TALKING:
                    self.is_talking = True
                    self.open_close_cycle_count = 0
                    self.talking_inactivity_counter = 0
                    print(f'[Mouth] TALKING FLAGGED — cycles threshold reached')
            elif self.open_frame_count > 0:
                print(f'[Mouth] Dismissed — only open for {self.open_frame_count} frames (min: {self.MIN_OPEN_FRAMES_PER_CYCLE})')

            self.open_frame_count = 0

        if self.is_talking:
            if cycle_completed:
                self.talking_inactivity_counter = 0
            else:
                self.talking_inactivity_counter += 1

            if self.talking_inactivity_counter >= self.TALKING_INACTIVITY_THRESHOLD:
                self.is_talking = False
                self.talking_inactivity_counter = 0
                print(f'[Mouth] Talking episode ended — inactivity threshold reached')

        self.previous_mouth_state = self.current_mouth_state
        return self.is_talking

    def reset(self):
        self.open_frame_count            = 0
        self.open_close_cycle_count      = 0
        self.last_mar                    = 0.0
        self.current_mouth_state         = 0
        self.is_talking                  = False
        self.talking_inactivity_counter  = 0
