import time

import cv2
import carla

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import numpy as np
from leaderboard.utils.route_manipulation import interpolate_trajectory

from customized_utils import get_angle, visualize_route, norm_2d, get_bbox, angle_from_center_view_fov
import os
import math
import pathlib

class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.record_every_n_step = 5
        self.wall_start = time.time()
        self.initialized = False

        parent_folder = os.environ['SAVE_FOLDER']
        string = pathlib.Path(os.environ['ROUTES']).stem
        self.save_path = pathlib.Path(parent_folder) / string

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)
        self.initialized = True


        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = CarlaDataProvider.get_map()

        self.min_d = 10000
        self.offroad_d = 10000
        self.wronglane_d = 10000
        self.dev_dist = 0


        # hop_resolution = 0.1
        # _, route = interpolate_trajectory(self._world, self.trajectory, hop_resolution)
        # visualize_route(route)
        # self.transforms = []
        # for x in route:
        #     self.transforms.append(x[0])
        # print('len(self.transforms)', len(self.transforms))



    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': -0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': 0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    },
                # addition
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.0, 'y': 0.0, 'z': 100.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'map'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -6, 'y': 0.0, 'z': 3,
                    'roll': 0.0, 'pitch': -20.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_with_car'
                    },
                {
                    'type': 'sensor.other.radar',
                    'x': 2, 'y': 0.0, 'z': 1,
                    'roll': 0.0, 'pitch': 5.0, 'yaw': 0.0,
                    'horizontal_fov': 35, 'vertical_fov': 20, 'range': 20,
                    'id': 'radar_central'
                    }
                ]

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        return {
                'rgb': rgb,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'gps': gps,
                'speed': speed,
                'compass': compass
                }



    # def set_trajectory(self, trajectory):
    #     self.trajectory = trajectory

    def set_args(self, args):
        self.deviations_path = os.path.join(args.deviations_folder, 'deviations.txt')
        self.args = args


    def record_other_actor_info_for_causal_analysis(self, ego_control_and_speed_info):
        def get_loc_and_ori(agent):
            agent_tra = agent.get_transform()
            agent_loc = agent_tra.location
            agent_rot = agent_tra.rotation
            return agent_loc.x, agent_loc.y, agent_rot.yaw

        data_row = []
        if ego_control_and_speed_info:
            data_row += ego_control_and_speed_info

        x, y, yaw = get_loc_and_ori(self._vehicle)
        data_row += [x, y, yaw]

        other_actor_info_path = os.path.join(self.args.deviations_folder, 'other_actor_info.txt')

        actors = self._world.get_actors()
        vehicle_list = actors.filter('*vehicle*')
        pedestrian_list = actors.filter('*walker*')




        for i, pedestrian in enumerate(pedestrian_list):
            d_angle_norm = angle_from_center_view_fov(pedestrian, self._vehicle, fov=90)
            if d_angle_norm == 0:
                within_view = True
            else:
                within_view = False

            x, y, yaw = get_loc_and_ori(pedestrian)
            data_row.extend([x, y, yaw, within_view])

        for i, vehicle in enumerate(vehicle_list):
            if vehicle.id == self._vehicle.id:
                continue

            d_angle_norm = angle_from_center_view_fov(vehicle, self._vehicle, fov=90)
            if d_angle_norm == 0:
                within_view = True
            else:
                within_view = False

            x, y, yaw = get_loc_and_ori(vehicle)
            data_row.extend([x, y, yaw, within_view])

        with open(other_actor_info_path, 'a') as f_out:
            f_out.write(','.join([str(d) for d in data_row])+'\n')



    def gather_info(self, ego_control_and_speed_info=None):
        if self.step % 1 == 0:
            self.record_other_actor_info_for_causal_analysis(ego_control_and_speed_info)


        ego_bbox = get_bbox(self._vehicle)
        ego_front_bbox = ego_bbox[:2]


        actors = self._world.get_actors()
        vehicle_list = actors.filter('*vehicle*')
        pedestrian_list = actors.filter('*walker*')

        min_d = 10000
        for i, vehicle in enumerate(vehicle_list):
            if vehicle.id == self._vehicle.id:
                continue

            d_angle_norm = angle_from_center_view_fov(vehicle, self._vehicle, fov=90)
            if d_angle_norm == 0:
                other_bbox = get_bbox(vehicle)
                for other_b in other_bbox:
                    for ego_b in ego_bbox:
                        d = norm_2d(other_b, ego_b)
                        # print('vehicle', i, 'd', d)
                        min_d = np.min([min_d, d])


        for i, pedestrian in enumerate(pedestrian_list):
            d_angle_norm = angle_from_center_view_fov(pedestrian, self._vehicle, fov=90)
            if d_angle_norm == 0:
                pedestrian_location = pedestrian.get_transform().location
                for ego_b in ego_front_bbox:
                    d = norm_2d(pedestrian_location, ego_b)
                    # print('pedestrian', i, 'd', d)
                    min_d = np.min([min_d, d])


        if min_d < self.min_d:
            self.min_d = min_d
            with open(self.deviations_path, 'a') as f_out:
                f_out.write('min_d,'+str(self.min_d)+'\n')



        angle_th = 120

        current_location = CarlaDataProvider.get_location(self._vehicle)
        current_transform = CarlaDataProvider.get_transform(self._vehicle)
        current_waypoint = self._map.get_waypoint(current_location, project_to_road=False, lane_type=carla.LaneType.Any)

        lane_center_waypoint = self._map.get_waypoint(current_location, lane_type=carla.LaneType.Any)
        lane_center_transform = lane_center_waypoint.transform
        lane_center_location = lane_center_transform.location


        dev_dist = current_location.distance(lane_center_location)

        if dev_dist > self.dev_dist:
            self.dev_dist = dev_dist
            with open(self.deviations_path, 'a') as f_out:
                f_out.write('dev_dist,'+str(self.dev_dist)+'\n')




        # print(current_location, current_waypoint.lane_type, current_waypoint.is_junction)
        # print(lane_center_location, lane_center_waypoint.lane_type, lane_center_waypoint.is_junction)


        if current_waypoint and not current_waypoint.is_junction:

            ego_forward = current_transform.get_forward_vector()
            lane_forward = lane_center_transform.get_forward_vector()


            dev_angle = 2 * get_angle(lane_forward.x, lane_forward.y, ego_forward.x, ego_forward.y) / np.pi
            # print(lane_forward, ego_forward, dev_angle)
            if dev_angle > 1:
                dev_angle = 2 - dev_angle
            elif dev_angle < -1:
                dev_angle = (-1) * (2 + dev_angle)

            # carla map has opposite y axis
            dev_angle *= -1




            def get_d(coeff, dev_angle):
                if coeff < 0:
                    dev_angle = 1 - dev_angle
                elif coeff > 0:
                    dev_angle = dev_angle + 1


                # print(dev_angle, coeff)

                n = 1
                rv = lane_center_waypoint.transform.get_right_vector()
                new_loc = carla.Location(lane_center_location.x + n*coeff*rv.x, lane_center_location.y + n*coeff*rv.y, 0)

                new_wp = self._map.get_waypoint(new_loc,project_to_road=False, lane_type=carla.LaneType.Any)

                while new_wp and new_wp.lane_type in [carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Bidirectional] and np.abs(new_wp.transform.rotation.yaw - lane_center_waypoint.transform.rotation.yaw) < angle_th:
                    prev_loc = new_loc
                    n += 1
                    new_loc = carla.Location(lane_center_location.x + n*coeff*rv.x, lane_center_location.y + n*coeff*rv.y, 0)
                    new_wp = self._map.get_waypoint(new_loc,project_to_road=False, lane_type=carla.LaneType.Any)
                    # if new_wp:
                    #     print(n, new_wp.transform.rotation.yaw)

                d = new_loc.distance(current_location)
                d *= dev_angle
                # print(d, new_loc, current_location)


                with open(self.deviations_path, 'a') as f_out:
                    if (not new_wp) or (new_wp.lane_type not in [carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Bidirectional]):
                        if not new_wp:
                            s = 'None wp'
                        else:
                            s = new_wp.lane_type
                        # print('offroad_d', d, s, coeff)
                        # if new_wp:
                        #     print('lanetype', new_wp.lane_type)
                        if d < self.offroad_d:
                            self.offroad_d = d
                            with open(self.deviations_path, 'a') as f_out:
                                f_out.write('offroad_d,'+str(self.offroad_d)+'\n')
                    else:
                        with open(self.deviations_path, 'a') as f_out:
                            # print('wronglane_d', d, coeff)
                            if d < self.wronglane_d:
                                self.wronglane_d = d
                                f_out.write('wronglane_d,'+str(self.wronglane_d)+'\n')

            get_d(-0.2, dev_angle)
            get_d(0.2, dev_angle)
