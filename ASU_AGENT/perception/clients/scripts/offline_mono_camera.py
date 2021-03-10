import os
import sys
import cv2
import time
import glob
import numpy as np
import threading as th
from utils import clean_world
from utils import FrameHandler
from utils import preprocess_frame
from utils import save_offline_video
from utils import save_offline_frames



CARLA_LIB_PATH = "../../../../carla/dist/"
IMG_WIDTH = 800
IMG_HEIGHT = 600    
CAM_FOV = 90            # camera field of view
RUNTIME = 30*10         # seconds


try:
    sys.path.append(glob.glob(CARLA_LIB_PATH + "carla-*%d.%d-%s.egg"%(
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64"
    ))[0])

except IndexError:
    pass
import carla


actors_list = []
offline_frames = []
frames_shape = (IMG_HEIGHT, IMG_WIDTH, 4)
try:
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    blueprint_lib = world.get_blueprint_library()
    # -------------------------------------------
    car1_bp = blueprint_lib.filter("model3")[0]
    available_spawns = world.get_map().get_spawn_points()
    spawn_pt = np.random.choice(available_spawns)
    car1 = world.spawn_actor(car1_bp, spawn_pt)
    print("spawned car1 successfully!")
    car1.set_autopilot(True)
    actors_list.append(car1)
    # -------------------------------------------
    rgbacam_bp = blueprint_lib.find("sensor.camera.rgb")
    rgbacam_bp.set_attribute("image_size_x", f"{IMG_WIDTH}")
    rgbacam_bp.set_attribute("image_size_y", f"{IMG_HEIGHT}")
    rgbacam_bp.set_attribute("fov", f"{CAM_FOV}")
    cam1_location = carla.Location(x=0.3, y=0.0, z=1.75)
    cam1_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    spawn_pt = carla.Transform(cam1_location, cam1_rotation)
    camera1 = world.spawn_actor(rgbacam_bp, spawn_pt, attach_to=car1)
    print("spawned rgba-camera successfully!")
    actors_list.append(camera1)
    frame_handler = FrameHandler()
    frame_shape = (IMG_HEIGHT, IMG_WIDTH, 4)
    camera1.listen(lambda data: 
        offline_frames.append(preprocess_frame(data, frames_shape))
    )
    # -------------------------------------------
    time.sleep(RUNTIME)
    car1.set_autopilot(False)

finally:
    clean_world(actors_list)
    save_offline_video(offline_frames)

