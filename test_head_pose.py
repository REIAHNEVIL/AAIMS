import cv2
import mediapipe as mp
import numpy as np
from modules.head_pose import HeadPoseDetector

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing   = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

head_pose = HeadPoseDetector()
cap       = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image                 = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results               = face_mesh.process(image)
    image.flags.writeable = True
    image                 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            pitch, yaw, roll, nose_2d, nose_3d, rotation_vector, translation_vector = \
                head_pose.get_angles(face_landmarks, image_width, image_height)

            if pitch is None:
                continue

            direction = head_pose.get_direction(pitch, yaw)
            flagged   = head_pose.detect_head_turned(pitch, yaw)

            nose_point           = (int(nose_2d[0]), int(nose_2d[1]))
            nose_direction_point = (int(nose_2d[0] + yaw * 10), int(nose_2d[1] - pitch * 10))
            cv2.line(image, nose_point, nose_direction_point, (255, 0, 0), 3)

            label_color = (0, 0, 255) if flagged else (0, 255, 0)
            cv2.putText(image, direction,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, label_color, 2)

            cv2.putText(image, f'Pitch: {round(pitch, 2)}',
                        (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'Yaw: {round(yaw, 2)}',
                        (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'Roll: {round(roll, 2)}',
                        (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(image,
                        f'Off-screen frames: '
                        f'{head_pose.sustained_off_screen_frames}/'
                        f'{head_pose.FRAMES_TO_FLAG_HEAD_TURN}',
                        (20, 420), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            if flagged:
                cv2.putText(image, 'FLAGGED',
                            (20, 380), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3)

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
