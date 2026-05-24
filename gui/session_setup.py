import tkinter as tk
from tkinter import ttk, messagebox

class SessionSetupScreen(tk.Tk):

    def __init__(self, on_start):
        super().__init__()
        self.on_start = on_start

        self.title('AAIMS — Session Setup')
        self.geometry('480x340')
        self.resizable(False, False)
        self.configure(bg='#F5F5F5')

        tk.Label(self,
                 text='Session Setup',
                 font=('Arial', 12, 'bold'),
                 bg='#F5F5F5', fg='#1A1A1A').pack(pady=(20, 5))

        ttk.Separator(self, orient='horizontal').pack(
            fill='x', padx=30, pady=(0, 15))

        form = tk.Frame(self, bg='#F5F5F5')
        form.pack(padx=40, fill='x')

        tk.Label(form, text='Student ID / Name:',
                 font=('Arial', 10), bg='#F5F5F5',
                 anchor='w').grid(row=0, column=0, sticky='w', pady=8)
        self.student_id = tk.Entry(form, font=('Arial', 10),
                                   width=28, relief='solid', bd=1)
        self.student_id.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=8)

        tk.Label(form, text='Exam / Session Name:',
                 font=('Arial', 10), bg='#F5F5F5',
                 anchor='w').grid(row=1, column=0, sticky='w', pady=8)
        self.exam_name = tk.Entry(form, font=('Arial', 10),
                                  width=28, relief='solid', bd=1)
        self.exam_name.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=8)
        self.exam_name.insert(0, 'Midterm Examination')

        tk.Label(form, text='Workstation Number:',
                 font=('Arial', 10), bg='#F5F5F5',
                 anchor='w').grid(row=2, column=0, sticky='w', pady=8)
        self.workstation = tk.Entry(form, font=('Arial', 10),
                                    width=28, relief='solid', bd=1)
        self.workstation.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=8)
        self.workstation.insert(0, '1')

        ttk.Separator(self, orient='horizontal').pack(
            fill='x', padx=30, pady=15)

        tk.Button(self,
                  text='Start Baseline Calibration →',
                  font=('Arial', 10, 'bold'),
                  bg='#2C3E50', fg='white',
                  activebackground='#1A252F',
                  relief='flat', cursor='hand2',
                  pady=6, padx=20,
                  command=self._validate_and_start).pack()

    def _validate_and_start(self):
        if not self.student_id.get().strip():
            messagebox.showerror(
                'Missing Information',
                'Please enter the Student ID or Name.')
            return

        config = {
            'student_id':        self.student_id.get().strip(),
            'exam_name':         self.exam_name.get().strip(),
            'workstation':       self.workstation.get().strip(),
            'yaw_threshold':     10,
            'absence_threshold': 30
        }

        self.destroy()
        self.on_start(config)
