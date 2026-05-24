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
        self.geometry('820x500')
        self.resizable(False, False)
        self.configure(bg='#F5F5F5')

        tk.Label(self,
                 text='Post-Examination Integrity Log',
                 font=('Arial', 12, 'bold'),
                 bg='#F5F5F5', fg='#1A1A1A').pack(pady=(18, 2))

        tk.Label(self,
                 text=f"Student: {config.get('student_id', '')}   "
                      f"|   Exam: {config.get('exam_name', '')}",
                 font=('Arial', 9),
                 bg='#F5F5F5', fg='#444444').pack()

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=30, pady=10)

        self.summary_label = tk.Label(
            self, text='Loading...',
            font=('Arial', 9),
            bg='#F5F5F5', fg='#1A1A1A')
        self.summary_label.pack(pady=(0, 8))

        table_frame = tk.Frame(self)
        table_frame.pack(fill='both', expand=True, padx=30, pady=(0, 10))

        cols = ('Start Time', 'End Time', 'Duration', 'Description')
        self.tree = ttk.Treeview(
            table_frame, columns=cols, show='headings', height=12)

        self.tree.heading('Start Time',  text='Start Time')
        self.tree.heading('End Time',    text='End Time')
        self.tree.heading('Duration',    text='Duration')
        self.tree.heading('Description', text='Description')

        self.tree.column('Start Time',  width=145, anchor='center')
        self.tree.column('End Time',    width=145, anchor='center')
        self.tree.column('Duration',    width=80,  anchor='center')
        self.tree.column('Description', width=370, anchor='w')

        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.tree.tag_configure('flagged', background='#FFE0E0')

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
            session_log = pd.read_csv(self.processor.log_path)

            if session_log.empty:
                self.summary_label.configure(
                    text='No behavioral events recorded during this session.')
                return

            total_episodes       = len(session_log)
            head_turn_count      = len(session_log[session_log['event_type'] == 'HEAD_TURN'])
            face_absent_count    = len(session_log[session_log['event_type'] == 'FACE_ABSENT'])
            mouth_movement_count = len(session_log[session_log['event_type'] == 'MOUTH_OPEN'])

            self.summary_label.configure(
                text=f'Total Episodes: {total_episodes}   |   '
                     f'Head Turned: {head_turn_count}   |   '
                     f'Face Absent: {face_absent_count}   |   '
                     f'Mouth Movement: {mouth_movement_count}')

            for _, episode in session_log.iterrows():
                self.tree.insert(
                    '', 'end',
                    values=(
                        episode['start_time'],
                        episode['end_time'],
                        f"{episode['duration_seconds']}s",
                        episode['description']
                    ),
                    tags=('flagged',))

        except FileNotFoundError:
            self.summary_label.configure(text='No events logged for this session.')
        except Exception as e:
            self.summary_label.configure(text=f'Error loading log: {e}')

    def _export(self):
        try:
            desktop_directory = os.path.join(os.path.expanduser('~'), 'Desktop')
            filename = (
                f"AAIMS_Log_"
                f"{self.config.get('student_id', 'unknown')}_"
                f"{self.config.get('exam_name', '')}.csv"
            ).replace(' ', '_')
            destination_path = os.path.join(desktop_directory, filename)
            shutil.copy(self.processor.log_path, destination_path)
            messagebox.showinfo('Export Successful',
                                f'Log exported to Desktop:\n{filename}')
        except Exception as e:
            messagebox.showerror('Export Failed', str(e))
