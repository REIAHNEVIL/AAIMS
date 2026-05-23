# main.py
# Entry point for AAIMS.
# Launches all screens in order and connects them together.

from modules.aggregator import AaimsProcessor
from gui.notice_screen       import NoticeScreen
from gui.session_setup       import SessionSetupScreen
from gui.calibration_screen  import CalibrationScreen
from gui.proctor_dashboard   import ProctorDashboard

# load all models once at startup
processor = AaimsProcessor()

def launch_notice():
    app = NoticeScreen(on_acknowledge=launch_session_setup)
    app.mainloop()

def launch_session_setup():
    app = SessionSetupScreen(on_start=launch_calibration)
    app.mainloop()

def launch_calibration(config):
    app = CalibrationScreen(
        processor=processor,
        config=config,
        on_done=launch_dashboard
    )
    app.mainloop()

def launch_dashboard(config):
    app = ProctorDashboard(
        processor=processor,
        config=config
    )
    app.mainloop()

if __name__ == '__main__':
    launch_notice()