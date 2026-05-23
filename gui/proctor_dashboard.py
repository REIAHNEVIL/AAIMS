# gui/proctor_dashboard.py
# Live monitoring dashboard for the proctor.
# Shows real-time detection status and alerts during the exam.

import tkinter as tk
from tkinter import ttk
import cv2
import threading
from PIL import Image, ImageTk

class ProctorDashboard(tk.Tk):

    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor
        self.config    = config
        self.processor.alert_callback = self._show_alert

        self.title(
            f"AAIMS — Monitoring: "
            f"{config.get('student_id', '')} | "
            f"{config.get('exam_name', '')}")
        self.geometry('780x520')
        self.resizable(False, False)
        self.configure(bg='#F5F5F5')

        # top info bar
        info = tk.Frame(self, bg='#2C3E50', pady=6)
        info.pack(fill='x')
        tk.Label(info,
                 text=f"Student: {config.get('student_id', '')}   |   "
                      f"Exam: {config.get('exam_name', '')}   |   "
                      f"Workstation: {config.get('workstation', '')}",
                 font=('Arial', 9), bg='#2C3E50',
                 fg='white').pack()

        # main content
        content = tk.Frame(self, bg='#F5F5F5')
        content.pack(fill='both', expand=True, padx=15, pady=10)

        # left: webcam feed
        left = tk.Frame(content, bg='#F5F5F5')
        left.pack(side='left', fill='both', expand=True)

        tk.Label(left, text='Live Feed',
                 font=('Arial', 9, 'bold'),
                 bg='#F5F5F5', fg='#444444').pack(anchor='w')

        self.video_label = tk.Label(left, bg='black')
        self.video_label.pack()

        # right: status panel
        right = tk.Frame(content, bg='#F5F5F5', width=230)
        right.pack(side='right', fill='y', padx=(12, 0))
        right.pack_propagate(False)

        tk.Label(right, text='Detection Status',
                 font=('Arial', 9, 'bold'),
                 bg='#F5F5F5', fg='#444444').pack(
                 anchor='w', pady=(0, 6))

        # status indicators
        self.indicators = {}
        for label, key in [
            ('Face Presence',   'face_present'),
            ('Head Pose',       'head_flag'),
            ('Face Absence',    'absence_flag'),
            ('Mouth Movement',  'mouth_alert_active'),
        ]:
            row = tk.Frame(right, bg='#EFEFEF',
                           pady=6, padx=8)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label,
                     font=('Arial', 9), bg='#EFEFEF',
                     fg='#1A1A1A').pack(side='left')
            dot = tk.Label(row, text='\u25cf',
                           font=('Arial', 14),
                           bg='#EFEFEF', fg='green')
            dot.pack(side='right')
            self.indicators[key] = dot

        # metrics
        ttk.Separator(right, orient='horizontal').pack(
            fill='x', pady=8)

        tk.Label(right, text='Metrics',
                 font=('Arial', 9, 'bold'),
                 bg='#F5F5F5', fg='#444444').pack(anchor='w')

        self.metric_labels = {}
        for label, key in [
            ('Direction', 'direction'),
            ('Yaw',       'yaw'),
            ('Pitch',     'pitch'),
            ('MAR',       'mar'),
        ]:
            row = tk.Frame(right, bg='#F5F5F5')
            row.pack(fill='x', pady=1)
            tk.Label(row, text=f'{label}:',
                     font=('Arial', 9), bg='#F5F5F5',
                     fg='#444444', width=9,
                     anchor='w').pack(side='left')
            val = tk.Label(row, text='---',
                           font=('Courier', 9),
                           bg='#F5F5F5', fg='#1A1A1A')
            val.pack(side='left')
            self.metric_labels[key] = val

        # alert label
        ttk.Separator(right, orient='horizontal').pack(
            fill='x', pady=8)

        self.alert_label = tk.Label(
            right, text='',
            font=('Arial', 9, 'bold'),
            bg='#F5F5F5', fg='red',
            wraplength=200, justify='center')
        self.alert_label.pack()

        # end session button
        tk.Button(right,
                  text='End Session',
                  font=('Arial', 9, 'bold'),
                  bg='#7B0000', fg='white',
                  activebackground='#5A0000',
                  relief='flat', cursor='hand2',
                  pady=5,
                  command=self._end_session).pack(
                  side='bottom', fill='x', pady=8)

        # start camera thread
        self.cap     = cv2.VideoCapture(0)
        self.running = True
        self.thread  = threading.Thread(
            target=self._camera_loop, daemon=True)
        self.thread.start()

    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            result = self.processor.process_frame(frame)
            self.after(0, self._update_gui, frame, result)

    def _update_gui(self, frame, result):
        # update webcam feed
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (480, 360))
        img   = ImageTk.PhotoImage(Image.fromarray(small))
        self.video_label.configure(image=img)
        self.video_label.image = img

        # update indicators
        for key, dot in self.indicators.items():
            val = result.get(key, 0)
            if key == 'face_present':
                dot.configure(
                    fg='green' if val else 'red')
            else:
                dot.configure(
                    fg='red' if val else 'green')

        # update metrics
        self.metric_labels['direction'].configure(
            text=result.get('direction', '---'))
        self.metric_labels['yaw'].configure(
            text=f"{result.get('yaw', 0):.1f} deg")
        self.metric_labels['pitch'].configure(
            text=f"{result.get('pitch', 0):.1f} deg")
        self.metric_labels['mar'].configure(
            text=f"{result.get('mar', 0):.4f}")

    def _show_alert(self, event_type):
        self.alert_label.configure(
            text=f'ALERT\n{event_type}')
        self.after(5000, lambda:
            self.alert_label.configure(text=''))

    def _end_session(self):
        self.running = False
        self.cap.release()
        self.destroy()

        from gui.integrity_log_screen import IntegrityLogScreen
        IntegrityLogScreen(
            self.processor, self.config).mainloop()