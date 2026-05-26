import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from frontend.styles import (
    BG_MAIN,
    TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_TITLE, FONT_SUBHEADING, FONT_SMALL,
    BTN_PRIMARY
)
from PIL import Image, ImageTk

CALIBRATION_FRAMES = 100

class CalibrationScreen(tk.Tk):

    def __init__(self, processor, config, on_done):
        super().__init__()
        self.processor  = processor
        self.config     = config
        self.on_done    = on_done

        self.title('AAIMS — Baseline Calibration')
        self.geometry('540x480')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        tk.Label(self, text='Baseline Calibration', font=FONT_TITLE, bg=BG_MAIN, fg=TEXT_PRIMARY).pack(pady=(12, 2))

        tk.Label(self, text='Sit naturally, face the camera, and keep your mouth closed.', font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack()

        self.video_label = tk.Label(self, bg='black', width=480, height=300)
        self.video_label.pack(pady=8)

        self.progress_label = tk.Label(self, text='Press Start when you are ready.', font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_PRIMARY)
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self, length=400, maximum=CALIBRATION_FRAMES)
        self.progress_bar.pack(pady=6)

        self.start_btn = tk.Button(self, text='Start Calibration', font=FONT_SUBHEADING, **BTN_PRIMARY, pady=5, padx=15,
            command=self._start)
        self.start_btn.pack(pady=6)

        self.cap = cv2.VideoCapture(0)
        self.calibrating = False
        self.angle_buffer = []
        self.mar_buffer = []
        self.captured = 0
        self.running = True

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._update_preview()


    def _update_preview(self):
        if not self.running:
            return

        success, frame = self.cap.read()
        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (480, 300))
            display_image = ImageTk.PhotoImage(Image.fromarray(resized_frame))
            self.video_label.configure(image=display_image)
            self.video_label.image = display_image

            if self.calibrating:
                self._process_calibration_frame(frame)

        self.after(33, self._update_preview)


    def _on_close(self):
        self.running = False
        self.cap.release()
        self.destroy()


    def _start(self):
        self.calibrating = True
        self.start_btn.configure(state='disabled')
        self.progress_label.configure(text='Calibrating — please stay still...')


    def _process_calibration_frame(self, frame):
        if self.captured >= CALIBRATION_FRAMES:
            return

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            angles = self.processor.head_pose.get_angles(landmarks, frame_width, frame_height)
            if angles[0] is not None:
                self.angle_buffer.append((angles[0], angles[1]))

            mouth_aspect_ratio = self.processor.mouth.compute_mar(landmarks.landmark, frame_width, frame_height)
            self.mar_buffer.append(mouth_aspect_ratio)

            self.captured = self.captured + 1
            self.progress_bar['value'] = self.captured
            self.progress_label.configure(text=f'Calibrating... {self.captured}/{CALIBRATION_FRAMES} frames')

        if self.captured >= CALIBRATION_FRAMES:
            self._finish()

    def _finish(self):
        self.processor.head_pose.calibrate(self.angle_buffer)

        if self.mar_buffer:
            self.processor.mouth.mar_baseline = float(np.mean(self.mar_buffer))

        print(f'Calibration complete.')
        print(f'Head baseline pitch (x): {self.processor.head_pose.pitch_baseline:.2f}')
        print(f'Head baseline yaw (y): {self.processor.head_pose.yaw_baseline:.2f}')
        print(f'MAR baseline: {self.processor.mouth.mar_baseline:.4f}')

        self.progress_label.configure(text='Calibration complete!', fg='green')

        self.running = False
        self.cap.release()

        self.after(1200, lambda: [
            self.destroy(),
            self.on_done(self.config)
        ])
