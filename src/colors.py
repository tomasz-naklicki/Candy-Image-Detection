from enum import Flag, auto
import numpy as np

# Defining the HSV color value ranges for each color
LYELLOW = np.array([10, 140, 145])
UYELLOW = np.array([30, 255, 255])
LPURPLE = np.array([66, 40, 0])
UPURPLE = np.array([175, 250, 160])
LRED = np.array([169, 35, 80])
URED = np.array([180, 255, 255])
LGREEN = np.array([33, 130, 0])
UGREEN = np.array([90, 255, 255])


class Color(Flag):
    Yellow = auto()
    Purple = auto()
    Red = auto()
    Green = auto()
