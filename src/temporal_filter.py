import time
from src.config import Config

class TemporalFilter:
    def __init__(self, confirmation_seconds=Config.CONFIRMATION_TIME_SEC):
        """Ensures fall pose persists continuously for X seconds before confirming trigger."""
        self.confirmation_seconds = confirmation_seconds
        self.fall_start_time = None

    def update(self, is_fall_detected):
        """
        Updates state timer.
        Returns: (is_confirmed_fall: bool, elapsed_seconds: float)
        """
        current_time = time.time()

        if is_fall_detected:
            if self.fall_start_time is None:
                self.fall_start_time = current_time
            
            elapsed = current_time - self.fall_start_time
            if elapsed >= self.confirmation_seconds:
                return True, elapsed
            return False, elapsed
        else:
            self.fall_start_time = None
            return False, 0.0