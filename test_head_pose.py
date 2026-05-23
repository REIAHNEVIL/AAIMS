# test_head_pose.py
# Tests the HeadPoseDetector module by replicating
# what the YouTube tutorial displays on screen.
# Also verifies that looking down is correctly detected
# since AAIMS exams are strictly on-screen only.

import cv2
import mediapipe as mp
import numpy as np
from modules.head_pose import HeadPoseDetector

# setup mediapipe face mesh same as the tutorial
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing   = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

head_pose = HeadPoseDetector()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image            = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results          = face_mesh.process(image)
    image.flags.writeable = True
    image            = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            x, y, z, nose_2d, nose_3d, rot_vec, trans_vec = head_pose.get_angles(
                face_landmarks, img_w, img_h
            )

            if x is None:
                continue

            # get direction from module
            direction = head_pose.get_direction(x, y)

            # check if this frame would be flagged
            flagged = head_pose.check_flag(x, y)

            # draw nose direction line
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # show direction label
            # turns red when flagged
            label_color = (0, 0, 255) if flagged else (0, 255, 0)
            cv2.putText(image, direction,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, label_color, 2)

            # show raw angle values
            cv2.putText(image, f'x: {round(x, 2)}',
                        (500, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.putText(image, f'y: {round(y, 2)}',
                        (500, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.putText(image, f'z: {round(z, 2)}',
                        (500, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

            # show flag counter so you can see it counting up
            cv2.putText(image, f'Flag counter: {head_pose.flag_counter}/{head_pose.FRAME_THRESHOLD}',
                        (20, 420), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            # show flagged status
            if flagged:
                cv2.putText(image, 'FLAGGED',
                            (20, 380), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3)

            # draw face mesh
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    cv2.imshow('Head Pose Module Test', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()