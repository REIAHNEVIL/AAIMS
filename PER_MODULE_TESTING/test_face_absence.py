import cv2
from modules.face_absence import FaceAbsenceDetector

detector = FaceAbsenceDetector()
cap      = cv2.VideoCapture(0)

print('Cover your camera or look away to test absence detection.')
print('Press ESC to quit.')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flagged         = detector.detect_absence(grayscale_frame)

    status       = 'FACE ABSENT' if detector.current_face_state == 0 else 'FACE PRESENT'
    status_color = (0, 0, 255) if detector.current_face_state == 0 else (0, 255, 0)

    cv2.putText(frame, status,
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 2)
    cv2.putText(frame,
                f'Absence counter: {detector.absence_counter}/{detector.ABSENCE_THRESHOLD}',
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if flagged:
        cv2.putText(frame, 'FLAGGED — FACE ABSENT TOO LONG',
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Face Absence Test', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
