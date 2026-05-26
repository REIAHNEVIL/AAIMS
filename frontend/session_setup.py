import tkinter as tk
from tkinter import ttk, messagebox
from frontend.styles import (
    BG_MAIN,
    TEXT_PRIMARY,
    FONT_TITLE, FONT_SUBHEADING, FONT_BODY,
    BTN_PRIMARY
)

class SessionSetupScreen(tk.Tk):

    def __init__(self, on_start):
        super().__init__()
        self.on_start = on_start

        self.title('AAIMS — Session Setup')
        self.geometry('480x340')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        tk.Label(self, text='Session Setup', font=FONT_TITLE, bg=BG_MAIN, fg=TEXT_PRIMARY).pack(pady=(20, 5))

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30, pady=(0, 15))

        form = tk.Frame(self, bg=BG_MAIN)
        form.pack(padx=40, fill='x')

        tk.Label(form, text='Student ID / Name:', font=FONT_BODY, bg=BG_MAIN, anchor='w').grid(row=0, column=0, sticky='w', pady=8)
        self.student_id = tk.Entry(form, font=FONT_BODY, width=28, relief='solid', bd=1)
        self.student_id.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=8)

        tk.Label(form, text='Exam / Session Name:', font=FONT_BODY, bg=BG_MAIN, anchor='w').grid(row=1, column=0, sticky='w', pady=8)
        self.exam_name = tk.Entry(form, font=FONT_BODY, width=28, relief='solid', bd=1)
        self.exam_name.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=8)
        self.exam_name.insert(0, 'Midterm Examination')

        tk.Label(form, text='Workstation Number:', font=FONT_BODY, bg=BG_MAIN, anchor='w').grid(row=2, column=0, sticky='w', pady=8)
        self.workstation = tk.Entry(form, font=FONT_BODY, width=28, relief='solid', bd=1)
        self.workstation.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=8)
        self.workstation.insert(0, '1')

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30, pady=15)

        tk.Button(self, text='Start Baseline Calibration →', font=FONT_SUBHEADING, **BTN_PRIMARY, pady=6, padx=20,
                  command=self._validate_and_start).pack()

    def _validate_and_start(self):
        if not self.student_id.get().strip():
            messagebox.showerror('Missing Information', 'Please enter the Student ID or Name.')
            return

        config = {
            'student_id': self.student_id.get().strip(),
            'exam_name': self.exam_name.get().strip(),
            'workstation': self.workstation.get().strip(),
            'yaw_threshold': 10,
            'absence_threshold': 30
        }

        print(f'[Session] Student: {config["student_id"]} | Exam: {config["exam_name"]} | Workstation: {config["workstation"]}')

        self.destroy()
        self.on_start(config)