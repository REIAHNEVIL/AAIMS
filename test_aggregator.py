# test_aggregator.py
# Tests the full AAIMS pipeline by running all three
# detection modules together on a live webcam feed.

import cv2
from modules.aggregator import AaimsProcessor

processor = AaimsProcessor()
cap       = cv2.VideoCapture(0)

print('Testing full pipeline. Press ESC to quit.')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    result = processor.process_frame(frame)

    # overall color — red if anything flagged, green if normal
    color = (0, 0, 255) if result['flagged'] else (0, 255, 0)

    # direction and head status
    cv2.putText(frame,
                f"Direction: {result['direction']}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    # mouth status
    cv2.putText(frame,
                f"MAR: {result['mar']:.4f} | "
                f"Mouth: {'OPEN' if result['mouth_flag'] else 'closed'}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    # mouth cycle counter
    cv2.putText(frame,
                f"Mouth cycles: {processor.mouth.cycle_counter}"
                f"/{processor.mouth.CYCLE_THRESHOLD}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    # face presence status
    cv2.putText(frame,
                f"Face: {'PRESENT' if result['face_present'] else 'ABSENT'}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    # absence counter
    cv2.putText(frame,
                f"Absence counter: {processor.face_absence.absence_counter}"
                f"/{processor.face_absence.ABSENCE_THRESHOLD}",
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    # mouth alert — stays visible for 5 seconds after flagging
    if result['mouth_alert_active']:
        cv2.putText(frame,
                    'MOUTH MOVEMENT DETECTED',
                    (20, 230), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    # general suspicious behavior alert
    if result['flagged']:
        cv2.putText(frame,
                    'SUSPICIOUS BEHAVIOR DETECTED',
                    (20, 270), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.imshow('AAIMS Pipeline Test', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f'Session log saved to: {processor.log_path}')