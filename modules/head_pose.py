import cv2
import numpy as np

class HeadPoseDetector:

    def __init__(self):
        # thresholds for flagging suspicious head movement
        # these match the ranges used in the reference tutorial
        self.yaw_threshold   = 10   # degrees left or right
        self.pitch_threshold = 10   # degrees up or down

        # baseline neutral angles set during calibration
        self.baseline_x = 0
        self.baseline_y = 0

        # consecutive frame counter for sustained movement
        self.flag_counter    = 0
        self.FRAME_THRESHOLD = 20   # ~0.67 seconds at 30fps

    def get_angles(self, face_landmarks, img_w, img_h):
        """
        Computes pitch (x), yaw (y), and roll (z) angles
        from MediaPipe face landmarks using solvePnP.
        Returns (x, y, z) angles or (None, None, None) if failed.
        """
        face_2d = []
        face_3d = []
        nose_2d = None
        nose_3d = None

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [1, 33, 61, 199, 263, 291]:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x_px = int(lm.x * img_w)
                y_px = int(lm.y * img_h)
                face_2d.append([x_px, y_px])
                face_3d.append([x_px, y_px, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # camera matrix using focal length approximation
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0,            img_h / 2],
            [0,            focal_length, img_w / 2],
            [0,            0,            1         ]
        ])

        # no lens distortion assumed
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # solve for rotation and translation vectors
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix
        )

        if not success:
            return None, None, None, None, None, None

        # convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # decompose rotation matrix into euler angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # scale angles (same as tutorial)
        x = angles[0] * 360  # pitch
        y = angles[1] * 360  # yaw
        z = angles[2] * 360  # roll

        return x, y, z, nose_2d, nose_3d, rot_vec, trans_vec

    def get_direction(self, x, y):
        """
        Returns a text label describing head direction.
        Left, right, and down are suspicious in AAIMS.
        Looking up is displayed but never flagged because
        the classroom has student assistants monitoring from above.
        """
        if y > self.yaw_threshold:
            return "Looking Left"
        elif y < -self.yaw_threshold:
            return "Looking Right"
        elif x < -self.pitch_threshold:
            return "Looking Down"
        elif x > self.pitch_threshold:
            return "Looking Up"
        else:
            return "Forward"

    def calibrate(self, angle_buffer):
        """
        Sets the neutral baseline from calibration frames.
        angle_buffer is a list of (x, y) tuples collected
        while the student sits in neutral position.
        """
        if not angle_buffer:
            return
        self.baseline_x = np.mean([a[0] for a in angle_buffer])
        self.baseline_y = np.mean([a[1] for a in angle_buffer])
        print(f'Head baseline — x: {self.baseline_x:.1f}  y: {self.baseline_y:.1f}')

    def check_flag(self, x, y):
        """
        Returns True if head has been turned beyond threshold
        for more than FRAME_THRESHOLD consecutive frames.
        
        Both horizontal (left/right) and downward movement are
        flagged because the exam is strictly on-screen only.
        Students should be looking at the monitor at all times.
        
        Looking up is excluded because the classroom setting has
        student assistants monitoring from above and notes placed
        above the monitor are not a realistic concern.
        """
        dx = x - self.baseline_x
        dy = abs(y - self.baseline_y)

        horizontal_turn = dy > self.yaw_threshold
        looking_down    = dx < -self.pitch_threshold

        if horizontal_turn or looking_down:
            self.flag_counter += 1
        else:
            self.flag_counter = 0

        return self.flag_counter >= self.FRAME_THRESHOLD