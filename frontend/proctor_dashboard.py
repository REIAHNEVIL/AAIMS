import tkinter as tk
from tkinter import ttk
import cv2
import threading
from PIL import Image, ImageTk
from frontend.styles import (
    BG_MAIN, BG_CARD, BG_HEADER,
    TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_LABEL, FONT_SMALL, FONT_DOT, FONT_MONO,
    BTN_DANGER
)

class ProctorDashboard(tk.Tk):

    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor
        self.config = config
        self.processor.alert_callback = self._show_alert

        self.title(
            f"AAIMS — Monitoring: "
            f"{config.get('student_id', '')} | "
            f"{config.get('exam_name', '')}")
        self.geometry('780x520')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        header_bar = tk.Frame(self, bg=BG_HEADER, pady=6)
        header_bar.pack(fill='x')
        tk.Label(header_bar, text=f"Student: {config.get('student_id', '')}   |   "  f"Exam: {config.get('exam_name', '')}   |   "  f"Workstation: {config.get('workstation', '')}", font=FONT_SMALL, bg=BG_HEADER, fg='white').pack()

        main_content = tk.Frame(self, bg=BG_MAIN)
        main_content.pack(fill='both', expand=True, padx=15, pady=10)

        webcam_panel = tk.Frame(main_content, bg=BG_MAIN)
        webcam_panel.pack(side='left', fill='both', expand=True)

        tk.Label(webcam_panel, text='Live Feed', font=FONT_LABEL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(anchor='w')

        self.video_label = tk.Label(webcam_panel, bg='black')
        self.video_label.pack()

        status_panel = tk.Frame(main_content, bg=BG_MAIN, width=230)
        status_panel.pack(side='right', fill='y', padx=(12, 0))
        status_panel.pack_propagate(False)

        tk.Label(status_panel, text='Detection Status', font=FONT_LABEL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(anchor='w', pady=(0, 6))

        self.indicators = {}
        for label, key in [
            ('Face Presence',  'face_present'),
            ('Head Pose',      'head_flag'),
            ('Face Absence',   'absence_flag'),
            ('Mouth Movement', 'mouth_alert_active'),
        ]:
            indicator_row = tk.Frame(status_panel, bg=BG_CARD, pady=6, padx=8)
            indicator_row.pack(fill='x', pady=2)
            tk.Label(indicator_row, text=label, font=FONT_SMALL, bg=BG_CARD, fg=TEXT_PRIMARY).pack(side='left')
            status_dot = tk.Label(indicator_row, text='●', font=FONT_DOT, bg=BG_CARD, fg='green')
            status_dot.pack(side='right')
            self.indicators[key] = status_dot

        ttk.Separator(status_panel, orient='horizontal').pack(fill='x', pady=8)

        tk.Label(status_panel, text='Metrics', font=FONT_LABEL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(anchor='w')

        self.metric_labels = {}
        for label, key in [
            ('Direction', 'direction'),
            ('Yaw',       'yaw'),
            ('Pitch',     'pitch'),
            ('MAR',       'mar'),
        ]:
            metric_row = tk.Frame(status_panel, bg=BG_MAIN)
            metric_row.pack(fill='x', pady=1)
            tk.Label(metric_row, text=f'{label}:', font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_SECONDARY, width=9, anchor='w').pack(side='left')
            metric_value_label = tk.Label(metric_row, text='---', font=FONT_MONO, bg=BG_MAIN, fg=TEXT_PRIMARY)
            metric_value_label.pack(side='left')
            self.metric_labels[key] = metric_value_label

        ttk.Separator(status_panel, orient='horizontal').pack(fill='x', pady=8)

        self.alert_label = tk.Label(status_panel, text='', font=FONT_LABEL, bg=BG_MAIN, fg='red', wraplength=200, justify='center')
        self.alert_label.pack()

        tk.Button(status_panel, text='End Session', font=FONT_LABEL, **BTN_DANGER, pady=5,
                  command=self._end_session).pack(side='bottom', fill='x', pady=8)

        self.cap = cv2.VideoCapture(0)
        self.running = True
        self._pending_after_ids = []
        self._gui_update_id = None
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        print(f'[Dashboard] Monitoring started — student: {config.get("student_id")} | exam: {config.get("exam_name")}')

    def _camera_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
            result = self.processor.process_frame(frame)
            
            try:
                self._gui_update_id = self.after(0, self._update_gui, frame, result)
            except tk.TclError:
                break

    def _update_gui(self, frame, result):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (480, 360))
        display_image = ImageTk.PhotoImage(Image.fromarray(resized_frame))
        self.video_label.configure(image=display_image)
        self.video_label.image = display_image

        for key, status_dot in self.indicators.items():
            current_value = result.get(key, 0)
            if key == 'face_present':
                status_dot.configure(fg='green' if current_value else 'red')
            else:
                status_dot.configure(fg='red' if current_value else 'green')

        self.metric_labels['direction'].configure(text=result.get('direction', '---'))
        self.metric_labels['yaw'].configure(text=f"{result.get('yaw', 0):.1f} deg")
        self.metric_labels['pitch'].configure(text=f"{result.get('pitch', 0):.1f} deg")
        self.metric_labels['mar'].configure(text=f"{result.get('mar', 0):.4f}")

    def _show_alert(self, event_type):
        print(f'[ALERT] {event_type}')
        self.alert_label.configure(text=f'ALERT\n{event_type}')
        after_id = self.after(5000, self._clear_alert)
        self._pending_after_ids.append(after_id)

    def _clear_alert(self):
        try:
            if self.winfo_exists():
                self.alert_label.configure(text='')
        except tk.TclError:
            pass

    def _end_session(self):
        print('[Dashboard] Session ended by proctor.')
        self.running = False
        if self._gui_update_id is not None:
            try:
                self.after_cancel(self._gui_update_id)
            except Exception:
                pass
        for after_id in self._pending_after_ids:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
        self.cap.release()
        self.destroy()

        from frontend.integrity_log_screen import IntegrityLogScreen
        IntegrityLogScreen(self.processor, self.config).mainloop()
