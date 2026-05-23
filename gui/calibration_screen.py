# gui/calibration_screen.py
# Baseline calibration screen for AAIMS.
# Shows live webcam feed so the student can position
# themselves correctly before calibration begins.

import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
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
        self.configure(bg='#F5F5F5')

        # instruction label at top
        tk.Label(self,
                 text='Baseline Calibration',
                 font=('Arial', 12, 'bold'),
                 bg='#F5F5F5', fg='#1A1A1A').pack(pady=(12, 2))

        tk.Label(self,
                 text='Sit naturally, face the camera, '
                      'and keep your mouth closed.',
                 font=('Arial', 9),
                 bg='#F5F5F5', fg='#444444').pack()

        # webcam feed — main focus of this screen
        self.video_label = tk.Label(self, bg='black',
                                    width=480, height=300)
        self.video_label.pack(pady=8)

        # progress label
        self.progress_label = tk.Label(
            self,
            text='Press Start when you are ready.',
            font=('Arial', 9),
            bg='#F5F5F5', fg='#1A1A1A')
        self.progress_label.pack()

        # progress bar
        self.progress_bar = ttk.Progressbar(
            self, length=400,
            maximum=CALIBRATION_FRAMES)
        self.progress_bar.pack(pady=6)

        # start button
        self.start_btn = tk.Button(
            self,
            text='Start Calibration',
            font=('Arial', 10, 'bold'),
            bg='#2C3E50', fg='white',
            activebackground='#1A252F',
            relief='flat', cursor='hand2',
            pady=5, padx=15,
            command=self._start)
        self.start_btn.pack(pady=6)

        # internal state
        self.cap          = cv2.VideoCapture(0)
        self.calibrating  = False
        self.angle_buffer = []
        self.mar_buffer   = []
        self.captured     = 0
        self.running      = True

        # mediapipe for landmark extraction during calibration
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # start showing webcam immediately
        self._update_preview()

    def _update_preview(self):
        """Updates the live webcam feed every 33ms (~30fps)."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            # convert and resize for display
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (480, 300))
            img   = ImageTk.PhotoImage(Image.fromarray(small))
            self.video_label.configure(image=img)
            self.video_label.image = img

            # process calibration if active
            if self.calibrating:
                self._process_calibration_frame(frame)

        self.after(33, self._update_preview)

    def _start(self):
        """Called when proctor clicks Start Calibration."""
        self.calibrating = True
        self.start_btn.configure(state='disabled')
        self.progress_label.configure(
            text='Calibrating — please stay still...')

    def _process_calibration_frame(self, frame):
        """Extracts landmarks from one calibration frame."""
        if self.captured >= CALIBRATION_FRAMES:
            return

        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # collect head pose angles
            angles = self.processor.head_pose.get_angles(
                landmarks, w, h)
            if angles[0] is not None:
                self.angle_buffer.append(
                    (angles[0], angles[1]))

            # collect MAR values
            mar = self.processor.mouth.compute_mar(
                landmarks.landmark, w, h)
            self.mar_buffer.append(mar)

            self.captured += 1
            self.progress_bar['value'] = self.captured
            self.progress_label.configure(
                text=f'Calibrating... '
                     f'{self.captured}/{CALIBRATION_FRAMES} frames')

        if self.captured >= CALIBRATION_FRAMES:
            self._finish()

    def _finish(self):
        """Saves baseline values and moves to dashboard."""
        # set head pose neutral baseline
        self.processor.head_pose.calibrate(self.angle_buffer)

        # set mouth resting MAR baseline
        if self.mar_buffer:
            self.processor.mouth.mar_baseline = float(
                np.mean(self.mar_buffer))

        print(f'Calibration complete.')
        print(f'Head baseline x: '
              f'{self.processor.head_pose.baseline_x:.2f}')
        print(f'Head baseline y: '
              f'{self.processor.head_pose.baseline_y:.2f}')
        print(f'MAR baseline: '
              f'{self.processor.mouth.mar_baseline:.4f}')

        self.progress_label.configure(
            text='Calibration complete!', fg='green')

        self.running = False
        self.cap.release()

        # short pause then proceed to dashboard
        self.after(1200, lambda: [
            self.destroy(),
            self.on_done(self.config)
        ])