import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from carla_project.src.common import CONVERTER, COLOR
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController


HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
DEBUG = int(os.environ.get('HAS_DISPLAY', 0))



def get_entry_point():
    return 'AutoPilot'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


class AutoPilot(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.save_path = None

        parent_folder = os.environ['SAVE_FOLDER']
        string = pathlib.Path(os.environ['ROUTES']).stem
        self.save_path = pathlib.Path(parent_folder) / string



    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _get_control(self, target, far_target, tick_data, _draw):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        _draw.text((5, 90), 'Speed: %.3f' % speed)
        _draw.text((5, 110), 'Target: %.3f' % target_speed)
        _draw.text((5, 130), 'Angle: %.3f' % angle_unnorm)
        _draw.text((5, 150), 'Angle Far: %.3f' % angle_far_unnorm)

        return steer, throttle, brake, target_speed

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        #     self._world.set_weather(WEATHERS[int(os.environ['WEATHER_INDEX'])])

        # if self.step % 100 == 0:
        #     index = (self.step // 100) % len(WEATHERS)
        #     self._world.set_weather(WEATHERS[index])

        data = self.tick(input_data)
        rgb_with_car = cv2.cvtColor(input_data['rgb_with_car'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        data['rgb_with_car'] = rgb_with_car

        topdown = data['topdown']
        rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))

        gps = self._get_position(data)

        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_topdown)

        _topdown.thumbnail((256, 256))
        _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))

        _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
        _draw = ImageDraw.Draw(_combined)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data, _draw)

        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
        _draw.text((5, 30), 'Steer: %.3f' % steer)
        _draw.text((5, 50), 'Throttle: %.3f' % throttle)
        _draw.text((5, 70), 'Brake: %s' % brake)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        # we only gether info every 2 frames for faster processing speed
        if self.step % 2 == 0:
            self.gather_info()


        # if this number is very small, we may not have the exact numbers and images for the event happening (e.g. the frame when a collision happen). However, this is usually ok if we only use these for retraining purpose
        record_every_n_steps = 3
        if self.step % record_every_n_steps == 0:
            self.save(record_every_n_steps, far_command, steer, throttle, brake, target_speed, data)
            self.save_json(record_every_n_steps, far_node, near_command, steer, throttle, brake, target_speed, data)

        return control


    def save_json(self, record_every_n_steps, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = int(self.step // record_every_n_steps)


        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # pos, , far_node, near_command
        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': near_command.value,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
        }

        pth = self.save_path / 'measurements'
        pth.mkdir(parents=False, exist_ok=True)
        (pth / ('%04d.json' % frame)).write_text(str(data))

    def save(self, record_every_n_steps, far_command, steer, throttle, brake, target_speed, tick_data):
        frame = int(self.step // record_every_n_steps)




        speed = tick_data['speed']
        string = os.environ['SAVE_FOLDER'] + '/' + pathlib.Path(os.environ['ROUTES']).stem

        center_str = string + '/' + 'rgb' + '/' + ('%04d.png' % frame)
        left_str = string + '/' + 'rgb_left' + '/' + ('%04d.png' % frame)
        right_str = string + '/' + 'rgb_right' + '/' + ('%04d.png' % frame)
        # topdown_str = string + '/' + 'topdown' + '/' + ('%04d.png' % frame)

        center = self.save_path / 'rgb' / ('%04d.png' % frame)
        left = self.save_path / 'rgb_left' / ('%04d.png' % frame)
        right = self.save_path / 'rgb_right' / ('%04d.png' % frame)
        topdown = self.save_path / 'topdown' / ('%04d.png' % frame)
        rgb_with_car = self.save_path / 'rgb_with_car' / ('%04d.png' % frame)

        data_row = ','.join([str(i) for i in [frame, far_command, speed, steer, throttle, brake, center_str, left_str, right_str]])
        with (self.save_path / 'measurements.csv').open("a") as f_out:
            f_out.write(data_row+'\n')

        Image.fromarray(tick_data['rgb']).save(center)
        Image.fromarray(tick_data['rgb_left']).save(left)
        Image.fromarray(tick_data['rgb_right']).save(right)
        # modification
        # Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]]).save(topdown)
        Image.fromarray(tick_data['topdown']).save(topdown)
        Image.fromarray(tick_data['rgb_with_car']).save(rgb_with_car)

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        return any(x is not None for x in [vehicle, light, walker])

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)

        self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        self._draw_line(p1, v1, z+2.5, (0, 0, 255))

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            self._draw_line(p2, v2, z+2.5)

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
        v1_hat = o1
        v1 = s1 * v1_hat

        self._draw_line(p1, v1, z+2.5, (255, 0, 0))

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            self._draw_line(p2, v2, z+2.5, (255, 0, 0))

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None
