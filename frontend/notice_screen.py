import tkinter as tk
from tkinter import ttk
from frontend.styles import (
    BG_MAIN, BG_CARD,
    TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_HEADING, FONT_SUBHEADING, FONT_SMALL,
    BTN_PRIMARY
)

class NoticeScreen(tk.Tk):

    def __init__(self, on_acknowledge):
        super().__init__()
        self.on_acknowledge = on_acknowledge

        self.title('AAIMS — Pre-Examination Notice')
        self.geometry('550x460')
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        tk.Label(self, text='Automated Academic Integrity Monitoring System', font=FONT_HEADING, bg=BG_MAIN, fg=TEXT_PRIMARY, wraplength=500).pack(pady=(20, 2))

        tk.Label(self, text='Pre-Examination Notice', font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_SECONDARY).pack(pady=(0, 10))

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30)

        notice_frame = tk.Frame(self, bg=BG_CARD)
        notice_frame.pack(fill='both', expand=True, padx=30, pady=10)

        notice = (
            'This workstation is currently being monitored by AAIMS.\n\n'
            'The following data will be collected and analyzed '
            'in real time during the examination:\n\n'
            '  •  Head orientation (pitch, yaw, roll angles)\n'
            '  •  Facial presence within the camera frame\n'
            '  •  Mouth movement patterns\n\n'
            'Important:\n'
            '  •  All processing is done locally on this device.\n'
            '  •  No video footage is recorded or transmitted.\n'
            '  •  Only behavioral event timestamps are logged.\n'
            '  •  Flagged events are reviewed by your instructor.\n\n'
            'By clicking Acknowledge, you confirm that you have '
            'read and understood this notice.'
        )

        tk.Label(notice_frame, text=notice, font=FONT_SMALL, bg=BG_CARD, fg=TEXT_PRIMARY, justify='left', wraplength=460).pack(padx=15, pady=12)

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30)

        tk.Button(self, text='I Acknowledge — Proceed', font=FONT_SUBHEADING, **BTN_PRIMARY, pady=8, padx=20,
                  command=self._acknowledge).pack(pady=12)

    def _acknowledge(self):
        self.destroy()
        self.on_acknowledge()
