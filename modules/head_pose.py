import cv2
import numpy as np

class HeadPoseDetector:

    def __init__(self):
        # thresholds for flagging suspicious head movement
        # these match the ranges used in the reference tutorial
        self.yaw_threshold   = 10   # degrees left or right
        self.pitch_threshold = 10   # degrees up or down

        # baseline neutral angles set during calibration
        self.pitch_baseline = 0
        self.yaw_baseline   = 0

        # consecutive frame counter for sustained off-screen movement
        self.sustained_off_screen_frames = 0
        self.FRAMES_TO_FLAG_HEAD_TURN    = 20   # ~0.67 seconds at 30fps

    def get_angles(self, face_landmarks, image_width, image_height):
        """
        Computes pitch, yaw, and roll angles from MediaPipe face
        landmarks using solvePnP.
        Returns (pitch, yaw, roll, nose_2d, nose_3d, rotation_vector,
        translation_vector) or (None, ...) if solvePnP failed.
        """
        face_2d = []
        face_3d = []
        nose_2d = None
        nose_3d = None

        for landmark_index, landmark in enumerate(face_landmarks.landmark):
            if landmark_index in [1, 33, 61, 199, 263, 291]:
                if landmark_index == 1:
                    nose_2d = (landmark.x * image_width, landmark.y * image_height)
                    nose_3d = (landmark.x * image_width, landmark.y * image_height,
                               landmark.z * 3000)

                x_pixel = int(landmark.x * image_width)
                y_pixel = int(landmark.y * image_height)
                face_2d.append([x_pixel, y_pixel])
                face_3d.append([x_pixel, y_pixel, landmark.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # camera matrix using focal length approximation
        focal_length = 1 * image_width
        camera_matrix = np.array([
            [focal_length, 0,            image_height / 2],
            [0,            focal_length, image_width  / 2],
            [0,            0,            1                ]
        ])

        # no lens distortion assumed
        distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

        # solve for rotation and translation vectors
        solved, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d, face_2d, camera_matrix, distortion_coefficients
        )

        if not solved:
            return None, None, None, None, None, None, None

        # convert rotation vector to rotation matrix, then decompose to euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        # scale angles (same as tutorial)
        pitch = euler_angles[0] * 360
        yaw   = euler_angles[1] * 360
        roll  = euler_angles[2] * 360

        return pitch, yaw, roll, nose_2d, nose_3d, rotation_vector, translation_vector

    def get_direction(self, pitch, yaw):
        """
        Returns a text label describing head direction.
        Left, right, and down are suspicious in AAIMS.
        Looking up is displayed but never flagged because
        the classroom has student assistants monitoring from above.
        """
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
        """
        Sets the neutral baseline from calibration frames.
        angle_buffer is a list of (pitch, yaw) tuples collected
        while the student sits in neutral position.
        """
        if not angle_buffer:
            return
        self.pitch_baseline = np.mean([a[0] for a in angle_buffer])
        self.yaw_baseline   = np.mean([a[1] for a in angle_buffer])
        print(f'Head baseline — pitch: {self.pitch_baseline:.1f}  yaw: {self.yaw_baseline:.1f}')

    def detect_head_turned(self, pitch, yaw):
        """
        Returns True if head has been turned beyond threshold
        for more than FRAMES_TO_FLAG_HEAD_TURN consecutive frames.

        Both horizontal (left/right) and downward movement are
        flagged because the exam is strictly on-screen only.
        Students should be looking at the monitor at all times.

        Looking up is excluded because the classroom setting has
        student assistants monitoring from above and notes placed
        above the monitor are not a realistic concern.
        """
        pitch_deviation = pitch - self.pitch_baseline
        yaw_deviation   = abs(yaw - self.yaw_baseline)

        horizontal_turn = yaw_deviation   > self.yaw_threshold
        looking_down    = pitch_deviation < -self.pitch_threshold

        if horizontal_turn or looking_down:
            self.sustained_off_screen_frames += 1
        else:
            self.sustained_off_screen_frames = 0

        return self.sustained_off_screen_frames >= self.FRAMES_TO_FLAG_HEAD_TURN
