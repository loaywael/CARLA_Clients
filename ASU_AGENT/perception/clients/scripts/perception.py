from dotenv import load_dotenv
import utils
import cv2 
import os

load_dotenv(verbose=True)
load_dotenv(dotenv_path="../../")
PROJECT_DIR = os.getenv("PROJECT_DIR")
SIMULATOR_DIR = os.getenv("SIMULATOR_DIR")
LEFT_FRAMES_DIR = PROJECT_DIR + "/clients/data/stereo_camera/left/"
RIGHT_FRAMES_DIR = PROJECT_DIR + "/clients/data/stereo_camera/right/"

utils.view_stereo(LEFT_FRAMES_DIR, RIGHT_FRAMES_DIR)