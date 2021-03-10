import os
import sys
import cv2
import time
import glob
import numpy as np
import threading as th
from dotenv import load_dotenv
import utils

load_dotenv(verbose=True)
load_dotenv(dotenv_path="../../")
PROJECT_DIR = os.getenv("PROJECT_DIR")
SIMULATOR_DIR = os.getenv("SIMULATOR_DIR")
CARLA_LIB_PATH = SIMULATOR_DIR+"/PythonAPI/carla/dist/"
LEFT_FRAMES_DIR = PROJECT_DIR + "/clients/data/stereo_camera/left/"
RIGHT_FRAMES_DIR = PROJECT_DIR + "/clients/data/stereo_camera/right/"
IMG_WIDTH = 800
IMG_HEIGHT = 600    
CAM_FOV = 90            # camera field of view
RUNTIME = 60            # seconds
TIMEOUT = 10

try:    # reqiored py3.7 
    sys.path.append(glob.glob(CARLA_LIB_PATH + "carla-*%d.%d-%s.egg"%(
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64"
    ))[0])

except IndexError as e:
    print(">>>[ERROR]---> ", e)

import carla


actors_list = []
# carla_debugger = carla.DebugHelper()
start_pt = carla.Transform(
    carla.Location(x=-77.887169, y=99.725639, z=0.700000), 
    carla.Rotation(pitch=0.0, yaw=-90.362541, roll=0.0)
)
try:
    # -------------- init world --------------
    print("initializing the world...!")
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    print("--- initialized world ---")
    # -------------- configure the car --------------
    blueprint_lib = world.get_blueprint_library()
    car_bp = blueprint_lib.filter("model3")[0]
    available_spawns = world.get_map().get_spawn_points()
    car_spawn_pt = np.random.choice(available_spawns)
    # -------------- configure the sensors --------------
    rgbacam_bp = blueprint_lib.find("sensor.camera.rgb")
    rgbacam_bp.set_attribute("image_size_x", f"{IMG_WIDTH}")
    rgbacam_bp.set_attribute("image_size_y", f"{IMG_HEIGHT}")
    rgbacam_bp.set_attribute("fov", f"{CAM_FOV}")
    leftcam_location = carla.Location(x=0.3, y=-0.15, z=1.75)
    rightcam_location = carla.Location(x=0.3, y=0.15, z=1.75)
    cam_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    leftcam_spawn_pt = carla.Transform(leftcam_location, cam_rotation)
    rightcam_spawn_pt = carla.Transform(rightcam_location, cam_rotation)
    
except Exception as e:
    print(">>>[ERROR]---> ", e)

finally:
    def main():
        # -------------- spwan the car --------------
        # carla_debugger.draw_point(start_pt, size=1, life_time=0)
        car = world.spawn_actor(car_bp, start_pt)
        print("spawned the car successfully!")
        actors_list.append(car)
        # -------------- spawn the sensors --------------
        left_camera = world.spawn_actor(
            rgbacam_bp, leftcam_spawn_pt, attach_to=car
        )
        right_camera = world.spawn_actor(
            rgbacam_bp, rightcam_spawn_pt, attach_to=car
        )
        print("spawned rgba-camera successfully!")
        actors_list.append(left_camera)
        actors_list.append(right_camera)
        # -------------- operate actors --------------
        car.set_autopilot(True)
        if (not left_camera.is_listening) and (not left_camera.is_listening):

            leftframe_handler = utils.FrameHandler(
                save2disk=True, frame_name='left', save_path=LEFT_FRAMES_DIR
            )
            rightframe_handler = utils.FrameHandler(
                save2disk=True, frame_name='right', save_path=RIGHT_FRAMES_DIR
            )
            left_camera.listen(leftframe_handler)
            right_camera.listen(rightframe_handler)

            # left_camera.listen(lambda data: data.save_to_disk(
            #         LEFT_FRAMES_DIR+"%06d.png"%data.frame_number
            #     )
            # )
            # right_camera.listen(lambda data: data.save_to_disk(
            #         RIGHT_FRAMES_DIR+"%06d.png"%data.frame_number
            #     )
            # )
    if __name__ == "__main__":
        stop_timer = th.Timer(RUNTIME, utils.clean_world, args=[actors_list])
        stop_timer.start()
        main()  
