import cv2
import mediapipe as mp
from modules.mouth_detection import MouthMovementDetector

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = MouthMovementDetector()
cap      = cv2.VideoCapture(0)

print('Open your mouth clearly to test detection.')
print('Press ESC to quit.')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame                    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results                      = face_mesh.process(rgb_frame)
    image_height, image_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            talking_detected = detector.detect_talking(landmarks, image_width, image_height)

            cv2.putText(frame, f'MAR: {detector.last_mar:.4f}',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2)

            mouth_state  = 'OPEN' if detector.current_mouth_state == 1 else 'CLOSED'
            state_color  = (0, 0, 255) if detector.current_mouth_state == 1 else (0, 255, 0)
            cv2.putText(frame, f'Mouth: {mouth_state}',
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, state_color, 2)

            cv2.putText(frame,
                        f'Open frames: {detector.open_frame_count}/{detector.MIN_OPEN_FRAMES_PER_CYCLE}',
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            if talking_detected:
                cv2.putText(frame, 'FLAGGED — MOUTH OPEN TOO LONG',
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

    cv2.imshow('Mouth Detection Test', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
