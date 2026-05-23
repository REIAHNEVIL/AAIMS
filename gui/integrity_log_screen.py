# gui/integrity_log_screen.py
# Post-exam integrity log review screen.
# Shows all flagged behavioral events from the session.

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import shutil
import os

class IntegrityLogScreen(tk.Tk):

    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor
        self.config    = config

        self.title('AAIMS — Session Integrity Log')
        self.geometry('680x480')
        self.resizable(False, False)
        self.configure(bg='#F5F5F5')

        # header
        tk.Label(self,
                 text='Post-Examination Integrity Log',
                 font=('Arial', 12, 'bold'),
                 bg='#F5F5F5', fg='#1A1A1A').pack(pady=(18, 2))

        tk.Label(self,
                 text=f"Student: {config.get('student_id', '')}   |   "
                      f"Exam: {config.get('exam_name', '')}",
                 font=('Arial', 9),
                 bg='#F5F5F5', fg='#444444').pack()

        ttk.Separator(self, orient='horizontal').pack(
            fill='x', padx=30, pady=10)

        # summary
        self.summary_label = tk.Label(
            self, text='Loading...',
            font=('Arial', 9),
            bg='#F5F5F5', fg='#1A1A1A')
        self.summary_label.pack(pady=(0, 8))

        # table
        table_frame = tk.Frame(self)
        table_frame.pack(fill='both', expand=True,
                         padx=30, pady=(0, 10))

        cols = ('Timestamp', 'Event Type')
        self.tree = ttk.Treeview(
            table_frame, columns=cols,
            show='headings', height=12)

        self.tree.heading('Timestamp',  text='Timestamp')
        self.tree.heading('Event Type', text='Event Type')
        self.tree.column('Timestamp',  width=220, anchor='center')
        self.tree.column('Event Type', width=380, anchor='center')

        scrollbar = ttk.Scrollbar(
            table_frame, orient='vertical',
            command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.tree.tag_configure('flagged', background='#FFE0E0')

        # buttons
        btn_frame = tk.Frame(self, bg='#F5F5F5')
        btn_frame.pack(fill='x', padx=30, pady=8)

        tk.Button(btn_frame,
                  text='Export Log to Desktop',
                  font=('Arial', 9, 'bold'),
                  bg='#2C3E50', fg='white',
                  relief='flat', cursor='hand2',
                  pady=5, padx=12,
                  command=self._export).pack(side='left')

        tk.Button(btn_frame,
                  text='Close',
                  font=('Arial', 9),
                  bg='#888888', fg='white',
                  relief='flat', cursor='hand2',
                  pady=5, padx=12,
                  command=self.destroy).pack(side='right')

        self._load_log()

    def _load_log(self):
        try:
            df = pd.read_csv(self.processor.log_path)
            total   = len(df)
            head    = len(df[df['event_type'] == 'HEAD_TURN'])
            absent  = len(df[df['event_type'] == 'FACE_ABSENT'])
            mouth   = len(df[df['event_type'] == 'MOUTH_OPEN'])

            self.summary_label.configure(
                text=f'Total Events: {total}   |   '
                     f'Head Turn: {head}   |   '
                     f'Face Absent: {absent}   |   '
                     f'Mouth Open: {mouth}')

            for _, row in df.iterrows():
                self.tree.insert(
                    '', 'end',
                    values=(row['timestamp'],
                            row['event_type']),
                    tags=('flagged',))

        except FileNotFoundError:
            self.summary_label.configure(
                text='No events logged for this session.')
        except Exception as e:
            self.summary_label.configure(
                text=f'Error loading log: {e}')

    def _export(self):
        try:
            desktop = os.path.join(
                os.path.expanduser('~'), 'Desktop')
            filename = (
                f"AAIMS_Log_"
                f"{self.config.get('student_id', 'unknown')}"
                f"_{self.config.get('exam_name', '')}.csv"
            ).replace(' ', '_')
            dest = os.path.join(desktop, filename)
            shutil.copy(self.processor.log_path, dest)
            messagebox.showinfo(
                'Export Successful',
                f'Log exported to Desktop:\n{filename}')
        except Exception as e:
            messagebox.showerror(
                'Export Failed', str(e))