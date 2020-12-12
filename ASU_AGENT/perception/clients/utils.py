import numpy as np
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