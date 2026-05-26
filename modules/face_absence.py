import cv2

class FaceAbsenceDetector:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.absence_counter   = 0
        self.ABSENCE_THRESHOLD = 30
        print('FaceAbsenceDetector loaded successfully.')

    def predict(self, grayscale_frame):
        faces = self.face_cascade.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(30, 30)
        )
        return 1 if len(faces) > 0 else 0

    def reset(self):
        self.absence_counter = 0
