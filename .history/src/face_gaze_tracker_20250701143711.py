import cv2
import numpy as np
import math

class FaceGazeTracker:
    def __init__(self):
        # ... your init code ...
        self.ring_angle = 0  # For animated rotation

    def draw_robo_eye(self, frame, center, radius=30):
        overlay = frame.copy()
        x, y = center
        r = radius

        # Base glow circle
        cv2.circle(overlay, center, r, (255, 255, 255), 2)
        cv2.circle(overlay, center, int(r * 0.6), (0, 255, 255), 1)

        # Glowing center pulse
        cv2.circle(overlay, center, 6, (0, 0, 255), -1)

        # Rotating ring dots
        for i in range(12):
            angle = math.radians(i * 30 + self.ring_angle)
            dx = int(r * 0.8 * math.cos(angle))
            dy = int(r * 0.8 * math.sin(angle))
            cv2.circle(overlay, (x + dx, y + dy), 2, (0, 255, 255), -1)

        # HUD style cross
        cv2.line(overlay, (x - 10, y), (x + 10, y), (255, 255, 255), 1)
        cv2.line(overlay, (x, y - 10), (x, y + 10), (255, 255, 255), 1)

        # Shadow for 3D effect
        cv2.circle(overlay, center, r + 3, (50, 50, 50), 1)

        # Transparent blend
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Animate ring angle
        self.ring_angle = (self.ring_angle + 5) % 360
