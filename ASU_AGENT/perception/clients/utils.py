import threading as th
import numpy as np
import glob
import cv2
import sys
import os
CARLA_LIB_PATH = "/home/loaywael/Development/ROS/carla/CARLA_0-2.9.10/PythonAPI/carla/dist/"

try:    # reqiored py3.7 
    sys.path.append(glob.glob(CARLA_LIB_PATH + "carla-*%d.%d-%s.egg"%(
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64"
    ))[0])

except IndexError as e:
    print("> ", e)

import carla

class FrameHandler(object):
    def __init__(self):
        self.after_sec = th.Timer(1, self.__calc_fps)
        self.img = None
        self.id = 0
        self.fps = 0
        self.counter = 0
        self.__avg_fps = []

    def __calc_fps(self):
        self.fps = self.counter
        self.__avg_fps.append(self.fps)
        self.counter = 0
    
    @property
    def avgfps(self):
        return round(sum(self.__avg_fps)/len(self.__avg_fps))

    def preprocess(self, frame):
        if not self.after_sec.is_alive():
            self.after_sec = th.Timer(1, self.__calc_fps)
            self.after_sec.start()
        img = np.array(frame.raw_data)
        img = img.reshape(600, 800, 4)
        self.img = img[:, :, :3]
        print(f"\rframe-id: {self.id}\t shape: {self.img.shape}\tfps: {self.fps}", end='')
        self.counter +=1
        self.id += 1


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


pth = "../../../ASU_AGENT/perception/detectors/ObjectDetection/yolov3/data/carla_data/rgb_frames/"
pack_yolo_dataset(pth)