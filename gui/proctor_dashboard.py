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
        header_bar = tk.Frame(self, bg='#2C3E50', pady=6)
        header_bar.pack(fill='x')
        tk.Label(header_bar,
                 text=f"Student: {config.get('student_id', '')}   |   "
                      f"Exam: {config.get('exam_name', '')}   |   "
                      f"Workstation: {config.get('workstation', '')}",
                 font=('Arial', 9), bg='#2C3E50',
                 fg='white').pack()

        # main content area
        main_content = tk.Frame(self, bg='#F5F5F5')
        main_content.pack(fill='both', expand=True, padx=15, pady=10)

        # left: webcam feed
        webcam_panel = tk.Frame(main_content, bg='#F5F5F5')
        webcam_panel.pack(side='left', fill='both', expand=True)

        tk.Label(webcam_panel, text='Live Feed',
                 font=('Arial', 9, 'bold'),
                 bg='#F5F5F5', fg='#444444').pack(anchor='w')

        self.video_label = tk.Label(webcam_panel, bg='black')
        self.video_label.pack()

        # right: status panel
        status_panel = tk.Frame(main_content, bg='#F5F5F5', width=230)
        status_panel.pack(side='right', fill='y', padx=(12, 0))
        status_panel.pack_propagate(False)

        tk.Label(status_panel, text='Detection Status',
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
            indicator_row = tk.Frame(status_panel, bg='#EFEFEF',
                                     pady=6, padx=8)
            indicator_row.pack(fill='x', pady=2)
            tk.Label(indicator_row, text=label,
                     font=('Arial', 9), bg='#EFEFEF',
                     fg='#1A1A1A').pack(side='left')
            status_dot = tk.Label(indicator_row, text='●',
                                  font=('Arial', 14),
                                  bg='#EFEFEF', fg='green')
            status_dot.pack(side='right')
            self.indicators[key] = status_dot

        # metrics
        ttk.Separator(status_panel, orient='horizontal').pack(
            fill='x', pady=8)

        tk.Label(status_panel, text='Metrics',
                 font=('Arial', 9, 'bold'),
                 bg='#F5F5F5', fg='#444444').pack(anchor='w')

        self.metric_labels = {}
        for label, key in [
            ('Direction', 'direction'),
            ('Yaw',       'yaw'),
            ('Pitch',     'pitch'),
            ('MAR',       'mar'),
        ]:
            metric_row = tk.Frame(status_panel, bg='#F5F5F5')
            metric_row.pack(fill='x', pady=1)
            tk.Label(metric_row, text=f'{label}:',
                     font=('Arial', 9), bg='#F5F5F5',
                     fg='#444444', width=9,
                     anchor='w').pack(side='left')
            metric_value_label = tk.Label(metric_row, text='---',
                                          font=('Courier', 9),
                                          bg='#F5F5F5', fg='#1A1A1A')
            metric_value_label.pack(side='left')
            self.metric_labels[key] = metric_value_label

        # alert label
        ttk.Separator(status_panel, orient='horizontal').pack(
            fill='x', pady=8)

        self.alert_label = tk.Label(
            status_panel, text='',
            font=('Arial', 9, 'bold'),
            bg='#F5F5F5', fg='red',
            wraplength=200, justify='center')
        self.alert_label.pack()

        # end session button
        tk.Button(status_panel,
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
            success, frame = self.cap.read()
            if not success:
                continue
            result = self.processor.process_frame(frame)
            try:
                self.after(0, self._update_gui, frame, result)
            except tk.TclError:
                break  # window was destroyed between the check and the after() call

    def _update_gui(self, frame, result):
        if not self.winfo_exists():
            return
        # update webcam feed
        rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (480, 360))
        display_image = ImageTk.PhotoImage(Image.fromarray(resized_frame))
        self.video_label.configure(image=display_image)
        self.video_label.image = display_image

        # update status indicators
        for key, status_dot in self.indicators.items():
            current_value = result.get(key, 0)
            if key == 'face_present':
                status_dot.configure(
                    fg='green' if current_value else 'red')
            else:
                status_dot.configure(
                    fg='red' if current_value else 'green')

        # update metric readouts
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
        self.after(5000, self._clear_alert)

    def _clear_alert(self):
        if self.winfo_exists():
            self.alert_label.configure(text='')

    def _end_session(self):
        self.running = False
        self.cap.release()
        self.destroy()

        from gui.integrity_log_screen import IntegrityLogScreen
        IntegrityLogScreen(
            self.processor, self.config).mainloop()
