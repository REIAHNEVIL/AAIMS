# gui/notice_screen.py
import tkinter as tk
from tkinter import ttk

class NoticeScreen(tk.Tk):

    def __init__(self, on_acknowledge):
        super().__init__()
        self.on_acknowledge = on_acknowledge

        self.title('AAIMS — Pre-Examination Notice')
        self.geometry('550x460')
        self.resizable(False, False)
        self.configure(bg='#F5F5F5')

        tk.Label(self,
                 text='Automated Academic Integrity Monitoring System',
                 font=('Arial', 11, 'bold'),
                 bg='#F5F5F5', fg='#1A1A1A',
                 wraplength=500).pack(pady=(20, 2))

        tk.Label(self,
                 text='Pre-Examination Notice',
                 font=('Arial', 10),
                 bg='#F5F5F5', fg='#444444').pack(pady=(0, 10))

        ttk.Separator(self, orient='horizontal').pack(
            fill='x', padx=30)

        frame = tk.Frame(self, bg='#EFEFEF')
        frame.pack(fill='both', expand=True, padx=30, pady=10)

        notice = (
            'This workstation is currently being monitored by AAIMS.\n\n'
            'The following data will be collected and analyzed '
            'in real time during the examination:\n\n'
            '  \u2022  Head orientation (pitch, yaw, roll angles)\n'
            '  \u2022  Facial presence within the camera frame\n'
            '  \u2022  Mouth movement patterns\n\n'
            'Important:\n'
            '  \u2022  All processing is done locally on this device.\n'
            '  \u2022  No video footage is recorded or transmitted.\n'
            '  \u2022  Only behavioral event timestamps are logged.\n'
            '  \u2022  Flagged events are reviewed by your instructor.\n\n'
            'By clicking Acknowledge, you confirm that you have '
            'read and understood this notice.'
        )

        tk.Label(frame, text=notice,
                 font=('Arial', 9),
                 bg='#EFEFEF', fg='#1A1A1A',
                 justify='left',
                 wraplength=460).pack(padx=15, pady=12)

        ttk.Separator(self, orient='horizontal').pack(
            fill='x', padx=30)

        tk.Button(self,
                  text='I Acknowledge — Proceed',
                  font=('Arial', 10, 'bold'),
                  bg='#2C3E50', fg='white',
                  activebackground='#1A252F',
                  relief='flat', cursor='hand2',
                  pady=8, padx=20,
                  command=self._acknowledge).pack(pady=12)

    def _acknowledge(self):
        self.destroy()
        self.on_acknowledge()