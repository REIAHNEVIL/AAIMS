import cv2
import os
import time

# Change this to 'face' or 'no_face' before running
LABEL               = 'no_face'
SAVE_DIRECTORY      = f'data/face_dataset/{LABEL}'
TARGET_FRAME_COUNT  = 900
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

cap = cv2.VideoCapture(1)
count     = 0
last_save = time.time()

print(f'Collecting {LABEL} images. Press Q to quit.')
print('Saving one image every 0.3 seconds...')

while True:
    success, frame = cap.read()
    if not success:
        break

    # ── AUTO STOP ──────────────────────────────────────
    if count >= TARGET_FRAME_COUNT:
        print(f'Target of {TARGET_FRAME_COUNT} images reached. Stopping.')
        break

    # Save a frame every 0.3 seconds
    if time.time() - last_save >= 0.3:
        filename     = os.path.join(SAVE_DIRECTORY, f'{LABEL}_{count:04d}.jpg')
        gray_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (128, 128))
        cv2.imwrite(filename, resized_frame)
        count    += 1
        last_save = time.time()
        print(f'  Saved: {count} images')

    cv2.putText(frame, f'Saved: {count} | Label: {LABEL}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Collecting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'Done! Collected {count} {LABEL} images.')
