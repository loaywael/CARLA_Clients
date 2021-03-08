import numpy as np
import glob
import cv2


class FrameHandler(object):
    def __init__(self):
        self.img = None
        self.id = 0

    def preprocess(self, frame, shape):
        img = np.array(frame.raw_data)
        img = img.reshape(shape)
        self.img = img[:, :, :3]
        print(f"frame-id: {self.id}\t frame-shape: {self.img.shape}\t")
        self.id += 1
        return self.img

# class FPS(object):
#     def __init__(self):
#         self.fps = 0
#         self.ready = False
    
#     def tick(self):
#         self.ready = True


def clean_world(actors_list):
    print("Attempting to clean all actors...!")
    for actor in actors_list[::-1]:
        actor.destroy()
    print("-"*11, "Cleaned All!", "-"*11)
   

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
            # cv2.imwrite(output_path+"%06d.png"%i, frame)
            print("-"*11, r"frame %06d"%i, "-"*11, end="")
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