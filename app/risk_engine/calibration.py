class Calibrator:
    def calibrate(self, raw_score):
        # Dummy isotonic/platt scaling calibration
        return min(max(raw_score * 0.95, 0.0), 1.0)
