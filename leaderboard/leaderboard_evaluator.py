#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import torchvision

# addition
import pathlib
import time
import logging

import carla
import signal
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

import atexit
from customized_utils import specify_args, start_server, exit_handler


# addition
WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,

        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.CloudySunset,

        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset,

        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.MidRainSunset,

        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.WetCloudySunset,

        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,

        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,

        # night modes
        # clear night
        carla.WeatherParameters(15.0, 0.0, 0.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
        # cloudy night
        carla.WeatherParameters(80.0, 0.0, 0.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
        # wet night
        carla.WeatherParameters(20.0, 0.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
        # midrain night
        carla.WeatherParameters(90.0, 0.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
        # wetcloudy night
        carla.WeatherParameters(80.0, 30.0, 50.0, 0.40, 0.0, -90.0, 0.0, 0.0, 0.0),
        # hardrain night
        carla.WeatherParameters(80.0, 60.0, 100.0, 1.00, 0.0, -90.0, 0.0, 0.0, 0.0),
        # softrain night
        carla.WeatherParameters(90.0, 15.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
]



sensors_to_icons = {
    'sensor.camera.semantic_segmentation':        'carla_camera',
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds

    # modification: 20.0 -> 10.0
    frame_rate = 10.0      # in Hz

    def __init__(self, args, statistics_manager, launch_server=False):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """

        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator.
        if os.environ['HAS_DISPLAY'] == '0':
            os.environ["DISPLAY"] = ''


        if launch_server:
            start_server(args.port)


        while True:
            try:
                self.client = carla.Client(args.host, int(args.port))
                break
            except:
                logging.exception("__init__ error")
                traceback.print_exc()




        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

        # addition
        parent_folder = args.save_folder
        print('-'*100, args.agent, os.environ['TEAM_AGENT'], '-'*100)
        if not os.path.exists(parent_folder):
            os.mkdir(parent_folder)
        string = pathlib.Path(os.environ['ROUTES']).stem
        current_record_folder = pathlib.Path(parent_folder) / string

        if os.path.exists(str(current_record_folder)):
            import shutil
            shutil.rmtree(current_record_folder)
        current_record_folder.mkdir(exist_ok=False)
        (current_record_folder / 'rgb').mkdir()
        (current_record_folder / 'rgb_left').mkdir()
        (current_record_folder / 'rgb_right').mkdir()
        (current_record_folder / 'topdown').mkdir()
        (current_record_folder / 'rgb_with_car').mkdir()
        # if args.agent == 'leaderboard/team_code/auto_pilot.py':
        #     (current_record_folder / 'topdown').mkdir()

        self.save_path = str(current_record_folder / 'events.txt')



    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)




    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model, vehicle.transform,                                    vehicle.rolename, color=vehicle.color, vehicle_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        while True:
            try:
                self.world = self.client.load_world(town)
                break
            except:
                logging.exception("_load_and_wait_for_world error")
                traceback.print_exc()

                start_server(args.port)
                self.client = carla.Client(args.host, int(args.port))

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
            """
            Computes and saved the simulation statistics
            """
            # register statistics
            current_stats_record = self.statistics_manager.compute_route_statistics(
                config,
                self.manager.scenario_duration_system,
                self.manager.scenario_duration_game,
                crash_message
            )

            print("\033[1m> Registering the route statistics\033[0m")
            self.statistics_manager.save_record(current_stats_record, config.index, self.save_path)
            self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_and_run_scenario(self, args, config, customized_data):
        """
        Load and run the scenario given by config.
        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"




        # addition: hack
        config.weather =  WEATHERS[args.weather_index]


        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")



        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
            # addition
            self.agent_instance.set_deviations_path(args.deviations_folder)
            config.agent = self.agent_instance

                        # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug, customized_data=customized_data)
            self.statistics_manager.set_scenario(scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario



        # addition
        # Set the appropriate road friction
        if config.friction is not None:
            friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            self.world.spawn_actor(friction_bp, transform)

        # night mode
        if config.weather.sun_altitude_angle < 0.0:
            for vehicle in scenario.ego_vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
            # addition: to turn on lights of
            actor_list = self.world.get_actors()
            vehicle_list = actor_list.filter('*vehicle*')
            for vehicle in vehicle_list:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

        try:

            print('start to run scenario')
            self.manager.run_scenario()
            print('stop to run scanario')
        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")

            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()



            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)

    def run(self, args, customized_data):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(self.save_path)
            self.statistics_manager.resume(self.save_path)
        else:
            self.statistics_manager.clear_record(self.save_path)
            route_indexer.save_state(args.checkpoint)

        while route_indexer.peek():
            # setup
            config = route_indexer.next()

            # run
            self._load_and_run_scenario(args, config, customized_data)


            route_indexer.save_state(self.save_path)

        # save global statistics
        # modification
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)


def main():
    arguments = specify_args()
    arguments.debug = True

    atexit.register(exit_handler, [arguments.port])

    statistics_manager = StatisticsManager()
    # 0, 1, 2, 3, 10, 11, 14, 15, 19
    # only 15 record vehicle's location for red light run




    # weather_indexes is a subset of integers in [0, 20]
    weather_indexes = [2]
    routes = [i for i in range(0, 10)]

    using_customized_route_and_scenario = False


    # multi_actors_scenarios = ['Scenario4']


    route_prefix = 'leaderboard/data/routes/route_'
    town_names = ['TownX']
    scenarios = ['ScenarioX']
    directions = ['directionX']

    # if we use autopilot, we only need one run of weather index since we constantly switching weathers for diversity
    if arguments.agent == 'leaderboard/team_code/auto_pilot.py':
        weather_indexes = weather_indexes[:1]

    time_start = time.time()

    for town_name in town_names:
        for scenario in scenarios:
            # if scenario not in multi_actors_scenarios:
            #     directions = [None]
            for dir in directions:
                for weather_index in weather_indexes:
                    arguments.weather_index = weather_index
                    os.environ['WEATHER_INDEX'] = str(weather_index)

                    if using_customized_route_and_scenario:

                        town_scenario_direction = town_name + '/' + scenario

                        folder_1 = os.environ['SAVE_FOLDER'] + '/' + town_name
                        folder_2 = folder_1 + '/' + scenario
                        if not os.path.exists(folder_1):
                            os.mkdir(folder_1)
                        if not os.path.exists(folder_2):
                            os.mkdir(folder_2)
                        if scenario in multi_actors_scenarios:
                            town_scenario_direction += '/' + dir
                            folder_2 += '/' + dir
                            if not os.path.exists(folder_2):
                                os.mkdir(folder_2)



                        os.environ['SAVE_FOLDER'] = folder_2
                        arguments.save_folder = os.environ['SAVE_FOLDER']

                        route_prefix = 'leaderboard/data/customized_routes/' + town_scenario_direction + '/route_'



                    for i, route in enumerate(routes):
                        print('\n'*10, route, '\n'*10)
                        customized_data = None


                        route_str = str(route)
                        if route < 10:
                            route_str = '0'+route_str
                        arguments.routes = route_prefix+route_str+'.xml'
                        os.environ['ROUTES'] = arguments.routes
                        if i == 0:
                            launch_server = True
                        else:
                            launch_server = False

                        try:
                            leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, launch_server)
                            leaderboard_evaluator.run(arguments, customized_data)

                        except Exception as e:
                            traceback.print_exc()
                        finally:
                            del leaderboard_evaluator
                        print('time elapsed :', time.time()-time_start)


if __name__ == '__main__':
    main()
