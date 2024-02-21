import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from yolov8 import VehicleTracker
from PIL import Image

yolo = VehicleTracker()

IM_WIDTH = 480
IM_HEIGHT = 360
Show_path_trajectory = False

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


def process_img_rgb(image, c) -> None:

    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    image = Image.fromarray(cv2.cvtColor(i3,cv2.COLOR_BGR2RGB))  
    r_image = yolo.track(image)
    i4 = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR)  

    cv2.imshow(c, i3)
    cv2.imshow('obj', i4)
    cv2.waitKey(1)
    return None


actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.load_world('Town05')
    debug = world.debug

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=2.5), carla.Rotation(pitch=-30))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_img_rgb(data, 'rgb'))


    time.sleep(120)


finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")