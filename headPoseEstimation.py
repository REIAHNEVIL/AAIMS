import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing   = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    frame_start = time.time()

    # Flip horizontally for selfie-view, then convert BGR → RGB
    image                 = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results               = face_mesh.process(image)
    image.flags.writeable = True
    image                 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark_index, landmark in enumerate(face_landmarks.landmark):
                if landmark_index in [1, 33, 61, 199, 263, 291]:
                    if landmark_index == 1:
                        nose_2d = (landmark.x * image_width,  landmark.y * image_height)
                        nose_3d = (landmark.x * image_width,  landmark.y * image_height,
                                   landmark.z * 3000)

                    x_pixel = int(landmark.x * image_width)
                    y_pixel = int(landmark.y * image_height)

                    face_2d.append([x_pixel, y_pixel])
                    face_3d.append([x_pixel, y_pixel, landmark.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length            = 1 * image_width
            camera_matrix           = np.array([
                [focal_length, 0,            image_height / 2],
                [0,            focal_length, image_width  / 2],
                [0,            0,            1                ]
            ])
            distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

            solved, rotation_vector, translation_vector = cv2.solvePnP(
                face_3d, face_2d, camera_matrix, distortion_coefficients)

            rotation_matrix, _    = cv2.Rodrigues(rotation_vector)
            euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch = euler_angles[0] * 360
            yaw   = euler_angles[1] * 360
            roll  = euler_angles[2] * 360

            if yaw < -10:
                direction_text = "Looking Left"
            elif yaw > 10:
                direction_text = "Looking Right"
            elif pitch < -10:
                direction_text = "Looking Down"
            elif pitch > 10:
                direction_text = "Looking Up"
            else:
                direction_text = "Forward"

            # Display nose direction line
            nose_projection, _ = cv2.projectPoints(
                nose_3d, rotation_vector, translation_vector,
                camera_matrix, distortion_coefficients)

            nose_point           = (int(nose_2d[0]), int(nose_2d[1]))
            nose_direction_point = (int(nose_2d[0] + yaw * 10), int(nose_2d[1] - pitch * 10))
            cv2.line(image, nose_point, nose_direction_point, (255, 0, 0), 3)

            cv2.putText(image, direction_text,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, f'Pitch: {np.round(pitch, 2)}',
                        (500, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'Yaw: {np.round(yaw, 2)}',
                        (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'Roll: {np.round(roll, 2)}',
                        (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_time = time.time() - frame_start
        fps        = 1 / frame_time if frame_time > 0 else 0
        cv2.putText(image, f'FPS: {int(fps)}',
                    (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
