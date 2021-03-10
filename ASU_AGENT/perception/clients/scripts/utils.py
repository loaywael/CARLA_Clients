import os
import sys
import cv2
import glob
import numpy as np
import threading as th
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv(verbose=True)
load_dotenv(dotenv_path="../../")
PROJECT_DIR = os.getenv("PROJECT_DIR")
SIMULATOR_DIR = os.getenv("SIMULATOR_DIR")
CARLA_LIB_PATH = SIMULATOR_DIR+"/PythonAPI/carla/dist/"

try:    # reqiored py3.7 
    sys.path.append(glob.glob(CARLA_LIB_PATH + "carla-*%d.%d-%s.egg"%(
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64"
    ))[0])

except IndexError as e:
    print(">>>[ERROR]---> ", e)

import carla

class FrameHandler(object):
    def __init__(self, save2disk=False, frame_name="", save_path=None):
        self.__id = 0
        self.__fps = 0
        self.__counter = 0
        self.__avg_fps = []
        self.frame = None
        self.__fname = frame_name
        self.__save2disk = save2disk
        self.__after_sec = th.Timer(1, self.__calc_fps)
        if not save_path:
            self.__save_path = PROJECT_DIR + "/clients/data/tmp/"
        else:
            self.__save_path = save_path
        os.makedirs(self.__save_path, exist_ok=True)

    def __calc_fps(self):
        self.__fps = self.__counter
        self.__avg_fps.append(self.__fps)
        self.__counter = 0
        print("\rframe-id: %i\t shape: %s\tfps: %i"%
        (self.__id, str(self.frame.shape), self.__fps), end=''
        )
    
    @property
    def avgfps(self):
        return round(sum(self.__avg_fps)/len(self.__avg_fps))

    def loggit(self):
        if not self.__after_sec.is_alive():
            self.__after_sec = th.Timer(1, self.__calc_fps)
            self.__after_sec.start()

    def _preprocess_frame(self, frame):
        img = np.array(frame.raw_data)
        img = img.reshape(600, 800, 4)
        self.frame = img[:, :, :3]
        
    def __call__(self, frame):
        self._preprocess_frame(frame)
        self.__counter +=1
        self.__id += 1
        if not self.__save2disk:
            self.loggit()
        else:
            name = self.__save_path+self.__fname+"_%06d.png"%self.__id
            plt.imwrite
            print("\rsaved frame: %s_%06d.png"%(self.__fname, self.__id))

class PlotTrajectory:
    def __init__(self, trajectory=None):
        self.__trajectory = trajectory
        self.x
    
    def plot_xy_plane(self):
        pass
    
    def plot_xyz_plane(self):
        pass

def view_stereo(left_path, right_path):
    left_frames = sorted(glob.glob(left_path+"/*"))
    right_frames = sorted(glob.glob(right_path+'/*'))
    print("found %i left & %i frames right"%(len(left_frames), len(right_frames)))
    pair_frames = list(zip(left_frames, right_frames))
    for left_img, right_img in pair_frames:
        left_img = cv2.cvtColor(plt.imread(left_img), cv2.COLOR_RGB2BGR)
        right_img = cv2.cvtColor(plt.imread(right_img), cv2.COLOR_RGBA2BGR)
        h, w = left_img.shape[:2]
        pair_board = np.hstack([left_img, np.zeros((h, 32, 3)), right_img])
        cv2.imshow("stereo_pair", pair_board)
        cv2.waitKey(25)
    cv2.destroyAllWindows()
        

def view_mono(frames_path):
    frames = sorted(glob.glob(frames_path+'/*'))
    print("found %i left & %i frames right"%(len(frames)))
    for frame in frames:
        key = cv2.waitKey(50)
        if key & 0xFF == ord('q'):
            break
        frame = cv2.cvtColor(plt.imread(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", frame)
    cv2.destroyAllWindows()


def clean_world(actors_list):
    print('\n', '\n', '-'*75)
    print("Attempting to clean all actors...!")
    no_of_actors = len(actors_list)
    if no_of_actors > 0:
        deleted = 0
        for actor in actors_list:
            if actor.is_alive:
                if isinstance(actor, (carla.libcarla.Vehicle,)):
                    actor.set_autopilot(False)
                actor.destroy()
                deleted += 1
        print("-"*11, f"cleaned {deleted}/{no_of_actors} actors!", "-"*11)
    else:
        print("--- already cleaned! ---")
   

def preprocess_frame(frame, shape):
    frame = np.array(frame.raw_data)
    frame = frame.reshape(shape)
    return frame[:, :, :3]


def save_offline_frames(offline_frames):
    print("saving offline frames...!")
    save_dir = "data/mono_camera/rgb_frames/"
    for i, frame in enumerate(offline_frames):
        cv2.imwrite(save_dir+"%06d.png"%i, frame)
    print("-"*11, "saved offline frames successfully!", "-"*11)


def save_offline_video(offline_frames, fps=20.0):
    # cv2.VideoWriter_fourcc(*"MP4V")
    fourcc =  0x7634706d  
    save_dir = "data/mono_camera/rgb_video/"
    shape = offline_frames[0].shape[1::-1]
    video_writer = cv2.VideoWriter(save_dir+"output.mp4", fourcc, fps, shape)
    print("saving offline video...!")
    for frame in offline_frames:
        video_writer.write(frame)
    video_writer.release()
    print("-"*11, "saved offline video successfully!", "-"*11)


def unpack_video_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    i = 0
    print("-"*11, "Unpacking frames...!", "-"*11)
    while True:
        ret, frame = cap.read()
        print(ret)
        if not ret:
            break
        else:
            cv2.imwrite(output_path+"%06d.png"%i, frame)
            print("-"*11, "\rframe %06d"%i, "-"*11)
            i += 1
    cap.release()


# src_pth = "data/mono_camera/rgb_video/output2.mp4"
# dst_pth = "data/mono_camera/rgb_frames/"
# cap = cv2.VideoCapture(src_pth)
# i = 0
# print("-"*11, "Unpacking frames", "-"*11)
# while True:
#     ret, frame = cap.read()
#     # print(ret)
#     if not ret:
#         print("\n", "-"*11, "Unpacking frames completed", "-"*11)
#         break

#     else:
#         cv2.imwrite(dst_pth+"%06d.png"%i, frame)
#         print("\r\t\tframe %06d"%i, end="")
#         i += 1
# print("\n")
# cap.release()

import json
import os

def pack_yolo_dataset(imgs_path):
    labels = {}
    with open("img_lbl.json", "w") as all_labels:
        for img_label_path in glob.glob(imgs_path+"*.txt"):
            if not img_label_path.endswith("classes.txt"):
                with open(img_label_path, "r") as img_label:
                    img_objects = img_label.readlines()
                    img_name = os.path.basename(img_label_path).split(".")[0]
                    labels[img_name] = img_objects
    print(labels)
