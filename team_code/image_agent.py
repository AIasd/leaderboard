import os
import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter

from team_code.base_agent import BaseAgent
from team_code.pid_controller import PIDController
# addition
import datetime
import pathlib

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))

# addition
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights
from carla_project.src.common import CONVERTER, COLOR
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    # modification

    # rgb = np.hstack((tick_data['rgb_left'], tick_data['rgb'], tick_data['rgb_right']))


    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144

    _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

    _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))

    _combined = _combined.resize((int(256 / _combined.size[1] * _combined.size[0]), 256))
    _topdown = Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]])
    _topdown.thumbnail((256, 256))
    _combined = Image.fromarray(np.hstack((_combined, _topdown)))


    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %s' % brake)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed)
    _draw.text((5, 110), 'Far Command: %s' % str(tick_data['far_command']))

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()


        # addition: modified from leaderboard/team_code/auto_pilot.py
        self.save_path = None

        parent_folder = os.environ['SAVE_FOLDER']
        string = pathlib.Path(os.environ['ROUTES']).stem
        self.save_path = pathlib.Path(parent_folder) / string




    # addition: modified from leaderboard/team_code/auto_pilot.py
    def save(self, steer, throttle, brake, tick_data):
        # frame = self.step // 10
        frame = self.step

        pos = self._get_position(tick_data)
        far_command = tick_data['far_command']
        speed = tick_data['speed']

        string = os.environ['SAVE_FOLDER']+'/'+pathlib.Path(os.environ['ROUTES']).stem



        center_str = string + '/' + 'rgb' + '/' + ('%04d.png' % frame)
        left_str = string + '/' + 'rgb_left' + '/' + ('%04d.png' % frame)
        right_str = string + '/' + 'rgb_right' + '/' + ('%04d.png' % frame)
        # topdown_str = string + '/' + 'topdown' + '/' + ('%04d.png' % frame)
        # rgb_with_car_str = string + '/' + 'rgb_with_car' + '/' + ('%04d.png' % frame)


        center = self.save_path / 'rgb' / ('%04d.png' % frame)
        left = self.save_path / 'rgb_left' / ('%04d.png' % frame)
        right = self.save_path / 'rgb_right' / ('%04d.png' % frame)
        topdown = self.save_path / 'topdown' / ('%04d.png' % frame)
        rgb_with_car = self.save_path / 'rgb_with_car' / ('%04d.png' % frame)

        data_row = ','.join([str(i) for i in [frame, far_command, speed, steer, throttle, brake, center_str, left_str, right_str]])
        with (self.save_path / 'measurements.csv' ).open("a") as f_out:
            f_out.write(data_row+'\n')


        Image.fromarray(tick_data['rgb']).save(center)
        Image.fromarray(tick_data['rgb_left']).save(left)
        Image.fromarray(tick_data['rgb_right']).save(right)

        # addition
        Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]]).save(topdown)
        Image.fromarray(tick_data['rgb_with_car']).save(rgb_with_car)


    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # addition:
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

    def tick(self, input_data):



        result = super().tick(input_data)
        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)


        rgb_with_car = cv2.cvtColor(input_data['rgb_with_car'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        result['rgb_with_car'] = rgb_with_car

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        # modification
        far_node, far_command = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target
        # addition:
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        result['far_command'] = far_command
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        result['topdown'] = topdown


        return result

    @torch.no_grad()
    def run_step_using_learned_controller(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        control = self.net.controller(points).cpu().squeeze()

        steer = control[0].item()
        desired_speed = control[1].item()
        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()




        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        points_cam = points.clone().cpu()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam).numpy()

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2

        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(tick_data, target_cam.squeeze(), points.cpu().squeeze(), steer, throttle, brake, desired_speed, self.step)

        # addition: from leaderboard/team_code/auto_pilot.py
        if self.step == 0:
            title_row = ','.join(['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake', 'center', 'left', 'right'])
            with (self.save_path / 'measurements.csv' ).open("a") as f_out:
                f_out.write(title_row+'\n')
        if self.step % 2 == 0:
            self.gather_info()

        record_every_n_steps = 3
        if self.step % record_every_n_steps == 0:
            self.save(steer, throttle, brake, tick_data)
        return control
