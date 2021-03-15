import os
import sys
import cv2
import glob
import numpy as np
import threading as th
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from mpl_toolkits import mplot3d


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
        self.frame = None
        self.__id = 0
        self.__fps = 0
        self.__counter = 0
        self.__avg_fps = []
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
    def __init__(self, trajectory=None, coordinates_type='camera'):
        self.__min_plot_dim = 1
        self.__coord_type = coordinates_type
        if isinstance(trajectory, (np.ndarray,)):
            if self.__coord_type == 'camera':
                self.__X = trajectory[2, :]
                self.__Y = trajectory[0, :]
                self.__Z = trajectory[1, :]    
        else:
            self.__X, self.__Y, self.__Z = [], [], []
        self.__xyz_plot, self.__xy_plot, self.__z_plot = self.__init_plot()
        
    @property
    def trajectory(self):
        return np.array(list(zip(self.__X, self.__Y, self.__Z)))

    def __init_plot(self):
        plt.style.use('ggplot')
        self.__fig = plt.figure(num=1, figsize=(15, 15), dpi=100)
        align_opt = dict(top=0.5, wspace=0.3, bottom=0.01)
        plots = gridspec.GridSpec(1, 2, figure=self.__fig, **align_opt)
        xy_z_plane = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=plots[0, 1])
        self.__xyz_plot = plt.subplot(plots[0, 0], projection='3d')
        self.__xy_plot = plt.subplot(xy_z_plane[:2, :])
        self.__z_plot = plt.subplot(xy_z_plane[2, :])
        self.__scale_axis()
        self.__init_labels()
        return self.__xyz_plot, self.__xy_plot, self.__z_plot
        
    def __init_labels(self):
        # ------- plot <xyz> plane -------
        self.__xyz_plot.set_xlabel('x-axis (forward/backward)')
        self.__xyz_plot.set_ylabel('y-axis (right/left)')
        self.__xyz_plot.set_zlabel('z-axis (up/down)')
        self.__xyz_plot.grid(True, alpha=0.3)
        # ------- plot <xy> plane -------
        self.__xy_plot.hlines(
            0, 
            -self.__min_plot_dim, 
            self.__min_plot_dim, 
            colors='gray', 
            linewidth=1, 
            alpha=0.5
        )
        self.__xy_plot.vlines(
            0, 
            -self.__min_plot_dim, 
            self.__min_plot_dim, 
            colors='gray',
            linewidth=1, 
            alpha=0.5
        )
        self.__xy_plot.set_xlabel('y-direction')
        self.__xy_plot.set_ylabel('x-direction')
        self.__xy_plot.grid(True, alpha=0.3)
        # ------- plot <z> plane -------
        self.__z_plot.hlines(
            0, 0, 
            self.__min_plot_dim, 
            colors='gray',
            linewidth=1,
            alpha=0.5
        )
        self.__z_plot.set_ylabel('z-direction')
        self.__z_plot.grid(True, alpha=0.3)

    def __clean_plot(self):
        self.__xyz_plot.clear()
        self.__xy_plot.clear()
        self.__z_plot.clear()
        
    def __scale_axis(self):
        if (len(self.__X) + len(self.__Y) + len(self.__Z)) > 0:
            max_dim = round(max(max(self.__X), max(self.__Y), max(self.__Z))) + len(self.__X)
#             if max_dim > self.__min_plot_dim:
            self.__clean_plot()
            self.__min_plot_dim = max_dim - (max_dim % 10) + 10
            self.__xyz_plot.set_xlim([-self.__min_plot_dim, self.__min_plot_dim])
            self.__xyz_plot.set_ylim([-self.__min_plot_dim, self.__min_plot_dim])
            self.__xyz_plot.set_zlim([0, self.__min_plot_dim])
            self.__xy_plot.set_xlim([-self.__min_plot_dim, self.__min_plot_dim])
            self.__xy_plot.set_ylim([-self.__min_plot_dim, self.__min_plot_dim])
            self.__z_plot.set_xlim([0, self.__min_plot_dim])
            self.__z_plot.set_ylim([-self.__min_plot_dim, self.__min_plot_dim])
        
    def plot(self):
        start_config = dict(c='g', s=75, marker='x', label='start point')
        stop_config = dict(c='b', s=75, marker='x', label='end point')
        line_config = dict(c='g', linestyle='--', linewidth=2, label='trajectory')
        # ---------- plot xyz view ----------
        self.__xyz_plot.scatter(self.__Y[0], self.__X[0], self.__Z[0], **start_config)
        self.__xyz_plot.plot(self.__Y, self.__X, self.__Z, **line_config)
        self.__xyz_plot.scatter(self.__Y[-1], self.__X[-1], self.__Z[-1], **stop_config)
        self.__xyz_plot.legend(facecolor='k')
        # ---------- plot xy plane ----------
        self.__xy_plot.scatter(self.__Y[0], self.__X[0], **start_config)
        self.__xy_plot.plot(self.__Y, self.__X, **line_config)
        self.__xy_plot.scatter(self.__Y[-1], self.__X[-1], **stop_config)
        self.__xy_plot.legend(facecolor='k')
        # ---------- plot z axis ----------
        self.__z_plot.plot(self.__Z, **line_config)
        self.__z_plot.legend(facecolor='k')
        plt.show()

    def plot_live(self, xyz_point):
        self.__X.append(xyz_point[2])
        self.__Y.append(xyz_point[0])
        self.__Z.append(xyz_point[1])
        self.__scale_axis()
        self.__init_labels()
        start_config = dict(c='g', s=75, marker='x', label='start point')
        stop_config = dict(c='b', s=75, marker='x', label='end point')
        line_config = dict(c='g', linestyle='--', linewidth=2, label='trajectory')
        # ---------- plot xyz view ----------
        self.__xyz_plot.scatter(self.__Y[0], self.__X[0] , self.__Z[0], **start_config)
        self.__xyz_plot.plot(self.__Y, self.__X, self.__Z, **line_config)
        self.__xyz_plot.scatter(self.__Y[-1], self.__X[-1], self.__Z[-1], **stop_config)
        self.__xyz_plot.legend(facecolor='k')
        # ---------- plot xy plane ----------
        self.__xy_plot.scatter(self.__Y[0], self.__X[0], **start_config)
        self.__xy_plot.plot(self.__Y, self.__X, **line_config)
        self.__xy_plot.scatter(self.__Y[-1], self.__X[-1], **stop_config)
        self.__xy_plot.legend(facecolor='k')
        # ---------- plot z axis ----------
        self.__z_plot.plot(self.__Z, **line_config)
        self.__z_plot.legend(facecolor='k')
    
    def __call__(self, xyz_point):
        if len(xyz_point) == 3:
            self.__current_point = xyz_point
#             animation.FuncAnimation(self.__fig, self.___plot_live)
        else:
            print("[ERROR]------> enter a valid 3d-vector point [x, y, z]")


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
