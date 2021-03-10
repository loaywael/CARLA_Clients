import os
import sys
import cv2
import time
import glob
import numpy as np
import threading as th
from dotenv import load_dotenv
import utils
from services.depth import depth_estimation

load_dotenv(verbose=True)
load_dotenv(dotenv_path="../../")
PROJECT_DIR = os.getenv("PROJECT_DIR")
SIMULATOR_DIR = os.getenv("SIMULATOR_DIR")
CARLA_LIB_PATH = SIMULATOR_DIR+"/PythonAPI/carla/dist/"
IMG_WIDTH = 800
IMG_HEIGHT = 600    
CAM_FOV = 90            # camera field of view
RUNTIME = 15            # seconds
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
        car = world.spawn_actor(car_bp, start_pt)
        print("spawned the car successfully!")
        actors_list.append(car)
        # -------------- spawn the sensors --------------
        left_camera = world.spawn_actor(rgbacam_bp, leftcam_spawn_pt, attach_to=car)
        right_camera = world.spawn_actor(rgbacam_bp, rightcam_spawn_pt, attach_to=car)
        print("spawned rgba-camera successfully!")
        actors_list.append(left_camera)
        actors_list.append(right_camera)
        # -------------- operate actors --------------
        car.set_autopilot(True)
        leftframe_handler = utils.FrameHandler()
        rightframe_handler = utils.FrameHandler()
        left_camera.listen(leftframe_handler)
        right_camera.listen(rightframe_handler)

        pleft = np.array([
            [640, 0, 640, 2176.0],
            [0, 480, 480, 552],
            [0, 0, 1, 1.4]
        ])
        pright = np.array([
            [640, 0, 640, 2176.0],
            [0, 480, 480, 552],
            [0, 0, 1, 1.4]
        ])
        try:
            while left_camera.is_listening and right_camera.is_listening:
                if (leftframe_handler.frame is not None) and (rightframe_handler.frame is not None):
                    # cv2.imshow("left", leftframe_handler.frame)
                    # cv2.imshow("right", rightframe_handler.frame)
                    estimator = depth_estimation.DepthEstimator(
                        leftframe_handler.frame, leftframe_handler.frame
                    )
                    cv2.imshow("disparity", estimator.comp_disparity())
                    key = cv2.waitKey(10)
                    if key & 0xFF == ord('q'):
                        break
            print('\n', "="*50)
            print(f"left-camera avg-fps: {leftframe_handler.avgfps}")
            print(f"right-camera avg-fps: {rightframe_handler.avgfps}")
        except Exception as e:
            print(">>>[ERROR]---> ", e)
            utils.clean_world(actors_list)

    if __name__ == "__main__":
        stop_timer = th.Timer(RUNTIME, utils.clean_world, args=[actors_list])
        stop_timer.start()
        main()
        cv2.destroyAllWindows()       
