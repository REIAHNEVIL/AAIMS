import cv2
import numpy as np

class HeadPoseDetector:

    def __init__(self):
        self.yaw_threshold   = 10
        self.pitch_threshold = 10

        self.pitch_baseline = 0
        self.yaw_baseline = 0

        self.sustained_off_screen_frames = 0
        self.FRAMES_TO_FLAG_HEAD_TURN = 20

    def get_angles(self, face_landmarks, image_width, image_height):
        face_2d = []  # 2D pixel positions of 6 key landmarks
        face_3d = []  # same landmarks with depth (z) from MediaPipe

        # step 1: extract 6 stable landmarks spread across the face
        for landmark_index, landmark in enumerate(face_landmarks.landmark):
            if landmark_index in [1, 33, 61, 199, 263, 291]:
                # convert normalized (0-1) coordinates to actual pixel positions
                x_pixel = int(landmark.x * image_width)
                y_pixel = int(landmark.y * image_height)
                face_2d.append([x_pixel, y_pixel])
                face_3d.append([x_pixel, y_pixel, landmark.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # step 2: build a simplified camera model
        # focal length is approximated as image width — standard shortcut without real calibration data
        focal_length  = image_width
        camera_matrix = np.array([
            [focal_length, 0,            image_height / 2],  # fx, 0,  cx
            [0,            focal_length, image_width  / 2],  # 0,  fy, cy
            [0,            0,            1                ]   # homogeneous row
        ])

        # assume no lens distortion
        distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

        # step 3: solve for the face's 3D rotation
        # solvePnP figures out what rotation explains the 2D landmark positions given the camera model
        solved, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d, face_2d, camera_matrix, distortion_coefficients
        )

        if not solved:
            return None, None, None, None, None

        # step 4: convert rotation vector (axis-angle) → rotation matrix (3x3)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # step 5: decompose rotation matrix → pitch, yaw, roll angles
        # raw output values are near 0, so multiply by 360 to get usable degree range
        euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = euler_angles[0] * 360
        yaw   = euler_angles[1] * 360
        roll  = euler_angles[2] * 360

        return pitch, yaw, roll, rotation_vector, translation_vector

    def get_direction(self, pitch, yaw):
        if yaw > self.yaw_threshold:
            return "Looking Left"
        elif yaw < -self.yaw_threshold:
            return "Looking Right"
        elif pitch < -self.pitch_threshold:
            return "Looking Down"
        elif pitch > self.pitch_threshold:
            return "Looking Up"
        else:
            return "Forward"

    def calibrate(self, angle_buffer):
        if not angle_buffer:
            return
        self.pitch_baseline = np.mean([a[0] for a in angle_buffer])
        self.yaw_baseline   = np.mean([a[1] for a in angle_buffer])
        print(f'Head baseline — pitch: {self.pitch_baseline:.1f}  yaw: {self.yaw_baseline:.1f}')

    def detect_head_turned(self, pitch, yaw):
        # measure how far the current angles deviate from the student's neutral position
        pitch_deviation = pitch - self.pitch_baseline
        yaw_deviation   = abs(yaw - self.yaw_baseline)

        horizontal_turn = yaw_deviation   > self.yaw_threshold
        looking_down    = pitch_deviation < -self.pitch_threshold

        if horizontal_turn or looking_down:
            self.sustained_off_screen_frames += 1
        else:
            self.sustained_off_screen_frames = 0

        # only flag after sustained off-screen movement, not a brief glance
        return self.sustained_off_screen_frames >= self.FRAMES_TO_FLAG_HEAD_TURN
