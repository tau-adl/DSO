from pyueye import ueye
import numpy as np
from UEYECamera import UEYECamera
import argparse
import cv2
import sys
import ctypes
import time


def recordVideo(parameterSetPath, fps, calib):

    camera = UEYECamera(parameterSetPath, fps)
    camera.setMemroyForVideoCapture()
    
    if int(calib) == 1:
        camera.captureCalibrationVideo()
    else:
        camera.captureVideo()
        
    del camera


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--parameterSetFile", help="path to parameterset")
    a.add_argument("--fps", help="wanted fps 30 or lower")
    a.add_argument("--calib", help="record for calibration or not 0 or 1")
    
    args = a.parse_args()
    recordVideo(args.parameterSetFile, args.fps, args.calib)