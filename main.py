import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

from modules.aggregator import AaimsProcessor
from frontend.notice_screen import NoticeScreen
from frontend.session_setup import SessionSetupScreen
from frontend.calibration_screen import CalibrationScreen
from frontend.proctor_dashboard import ProctorDashboard

processor = AaimsProcessor()

def launch_notice():
    print('[AAIMS] Starting — notice screen.')
    app = NoticeScreen(on_acknowledge=launch_session_setup)
    app.mainloop()

def launch_session_setup():
    print('[AAIMS] Notice acknowledged — session setup screen.')
    app = SessionSetupScreen(on_start=launch_calibration)
    app.mainloop()

def launch_calibration(config):
    print(f'[AAIMS] Session configured — launching calibration.')
    app = CalibrationScreen(processor=processor, config=config, on_done=launch_dashboard)
    app.mainloop()

def launch_dashboard(config):
    print(f'[AAIMS] Calibration done — launching dashboard.')
    app = ProctorDashboard(processor=processor, config=config)
    app.mainloop()

if __name__ == '__main__':
    launch_notice()
