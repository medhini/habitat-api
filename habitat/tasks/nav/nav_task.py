#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Type, Tuple

import attr
import cv2
import numpy as np
from gym import spaces
import lazy_property
import math
import quaternion
from typing import Union
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)

import habitat_sim.utils
from habitat.config import Config
from habitat.core.simulator import SimulatorActions
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, Measurements
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorSuite,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
from habitat.utils.visualizations import maps

COLLISION_PROXIMITY_TOLERANCE: float = 1e-3
MAP_THICKNESS_SCALAR: int = 1250

def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal that can be specified by object_id or position or object
    category.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius.
    """
    room_aabb: Tuple[float] = attr.ib(default=None, validator=not_none_validator)
    # room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: str = attr.ib(default=None, validator=not_none_validator)

class SE3:
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans

    def inv(self):
        rot_inv = self.rot.inverse()
        return SE3(quaternion.as_rotation_matrix(rot_inv), -self.trans)

@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[ShortestPathPoint]] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomNavigationEpisode(NavigationEpisode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[RoomGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
        
@registry.register_sensor
class EpisodicGPSAndCompassSensor(Sensor):
    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gps_and_compass"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        state = self._sim.get_agent_state()

        rot = quaternion.as_rotation_matrix(state.rotation)
        trans = state.position

        # look_dir = habitat_sim.utils.quat_rotate_vector(state.rotation, habitat_sim.geo.FRONT)
        # theta = np.array([np.rad2deg(np.arctan2(look_dir[0], -look_dir[2]))]).astype(np.float64)

        theta = np.array([np.rad2deg(np.arctan2(rot[0,2], rot[0,0]))]).astype(np.float64)
        trans = np.array([trans[0], trans[2]]).astype(np.float64)
        
        return np.concatenate([theta, trans]).astype(np.float64)

@registry.register_sensor
class AgentRotationSensor(Sensor):
    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_rotation"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,3),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        
        return quaternion.as_rotation_matrix(rotation_world_agent)

@registry.register_sensor
class AgentPositionSensor(Sensor):
    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        ref_position = agent_state.position
        
        return np.array([ref_position[0], ref_position[1], ref_position[2]]).astype(np.float64)

@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in the PointNav task.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self._goal_format == "CARTESIAN":
            sensor_shape = (3,)
        else:
            sensor_shape = (2,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        ref_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        direction_vector = (
            np.array(episode.goals[0].position, dtype=np.float32)
            - ref_position
        )
        direction_vector_agent = quaternion_rotate_vector(
            rotation_world_agent.inverse(), direction_vector
        )
        
        if self._goal_format == "POLAR":
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent


@registry.register_sensor
class RoomGoalSensor(Sensor):
    r"""Sensor for RoomGoal observations which are used in the RoomNav task.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the RoomGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the roomgoal is specified. Current options for goal format are
            cartesian and polar.

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self.room_name_to_id = {
            'bathroom': 0,
            'bedroom': 1,
            'dining room': 2,
            'kitchen': 3,
            'living room': 4 
        }

        # self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        # assert self._goal_format in ["CARTESIAN", "POLAR"]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "roomgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=min(self.room_name_to_id.values()),
            high=max(self.room_name_to_id.values()),
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(self, observations, episode):
        return np.array([self.room_name_to_id[episode.goals[0].room_name]])

@registry.register_sensor
class StaticPointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in the StaticPointNav
    task. For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.
    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        super().__init__(sim, config)
        self._initial_vector = None
        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "static_pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self._goal_format == "CARTESIAN":
            sensor_shape = (3,)
        else:
            sensor_shape = (2,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            # Only compute the direction vector when a new episode is started.
            self.current_episode_id = episode_id
            agent_state = self._sim.get_agent_state()
            ref_position = agent_state.position
            rotation_world_agent = agent_state.rotation

            direction_vector = (
                np.array(episode.goals[0].position, dtype=np.float32)
                - ref_position
            )
            direction_vector_agent = quaternion_rotate_vector(
                rotation_world_agent.inverse(), direction_vector
            )

            if self._goal_format == "POLAR":
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                direction_vector_agent = np.array(
                    [rho, -phi], dtype=np.float32
                )

            self._initial_vector = direction_vector_agent
        return self._initial_vector


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(
            rotation_world_agent.inverse(), direction_vector
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim, config):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float,
        )

    def get_observation(self, observations, episode):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )


@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(self, sim: Simulator, config: Config):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spl"

    def reset_metric(self, episode):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, action):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if (
            action == self._sim.index_stop_action
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            ep_success = 1

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(SimulatorActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot

@registry.register_measure
class OracleAction(Measure):
    r"""Utility class for extracting the action on the shortest path to the
        goal.
    Args:
        sim: HabitatSim instance.
        goal_radius: Distance between the agent and the goal for it to be
            considered successful.
        return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    """

    def __init__(
        self, sim: Simulator, config: Config):
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), "{} must have a method called geodesic_distance".format(
            type(sim).__name__
        )

        self._sim = sim
        self._config = config
        self._max_delta = self._sim.config.FORWARD_STEP_SIZE - 1e-6
        self._goal_radius = self._sim.config.FORWARD_STEP_SIZE
        self._step_size = self._sim.config.FORWARD_STEP_SIZE

        self._mode = (
            "geodesic_path"
            if getattr(sim, "get_straight_shortest_path_points", None)
            is not None
            else "greedy"
        )
        self._return_one_hot = False
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracleaction"
    
    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(self, goal_pos: np.array) -> Union[int, np.array]:
        """Returns the next action along the shortest path.
        """
        if (
            np.linalg.norm(goal_pos - self._sim.get_agent_state().position)
            <= self._goal_radius
        ):
            return self._get_return_value(SimulatorActions.STOP)

        max_grad_dir = self._est_max_grad_dir(goal_pos)
        if max_grad_dir is None:
            return self._get_return_value(SimulatorActions.MOVE_FORWARD)
        return self._step_along_grad(max_grad_dir)

    def _step_along_grad(
        self, grad_dir: np.quaternion
    ) -> Union[int, np.array]:
        current_state = self._sim.get_agent_state()
        alpha = angle_between_quaternions(grad_dir, current_state.rotation)
        if alpha <= np.deg2rad(self._sim.config.TURN_ANGLE) + 1e-6:
            return self._get_return_value(SimulatorActions.MOVE_FORWARD)
        else:
            sim_action = SimulatorActions.TURN_LEFT
            self._sim.step(sim_action)
            best_turn = (
                SimulatorActions.TURN_LEFT
                if (
                    angle_between_quaternions(
                        grad_dir, self._sim.get_agent_state().rotation
                    )
                    < alpha
                )
                else SimulatorActions.TURN_RIGHT
            )
            self._reset_agent_state(current_state)
            return self._get_return_value(best_turn)

    def _reset_agent_state(self, state: habitat_sim.AgentState) -> None:
        self._sim.set_agent_state(
            state.position, state.rotation, reset_sensors=False
        )

    def _geo_dist(self, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(
            self._sim.get_agent_state().position, goal_pos
        )

    def _est_max_grad_dir(self, goal_pos: np.array) -> np.array:
        current_state = self._sim.get_agent_state()
        current_pos = current_state.position

        if self.mode == "geodesic_path":
            points = self._sim.get_straight_shortest_path_points(
                self._sim.get_agent_state().position, goal_pos
            )
            # Add a little offset as things get weird if
            # points[1] - points[0] is anti-parallel with forward
            if len(points) < 2:
                return None
            max_grad_dir = quaternion_from_two_vectors(
                self._sim.forward_vector,
                points[1]
                - points[0]
                + 1e-6
                * np.cross(self._sim.up_vector, self._sim.forward_vector),
            )
            max_grad_dir.x = 0
            max_grad_dir = np.normalized(max_grad_dir)
        else:
            current_rotation = self._sim.get_agent_state().rotation
            current_dist = self._geo_dist(goal_pos)

            best_geodesic_delta = -2 * self._max_delta
            best_rotation = current_rotation
            for _ in range(0, 360, self._sim.config.TURN_ANGLE):
                sim_action = SimulatorActions.MOVE_FORWARD
                self._sim.step(sim_action)
                new_delta = current_dist - self._geo_dist(goal_pos)

                if new_delta > best_geodesic_delta:
                    best_rotation = self._sim.get_agent_state().rotation
                    best_geodesic_delta = new_delta

                # If the best delta is within (1 - cos(TURN_ANGLE))% of the
                # best delta (the step size), then we almost certainly have
                # found the max grad dir and should just exit
                if np.isclose(
                    best_geodesic_delta,
                    self._max_delta,
                    rtol=1 - np.cos(np.deg2rad(self._sim.config.TURN_ANGLE)),
                ):
                    break

                self._sim.set_agent_state(
                    current_pos,
                    self._sim.get_agent_state().rotation,
                    reset_sensors=False,
                )

                sim_action = SimulatorActions.TURN_LEFT
                self._sim.step(sim_action)

            self._reset_agent_state(current_state)

            max_grad_dir = best_rotation

        return max_grad_dir

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        r"""Sets the mode for how the greedy follower determines the best next
            step.
        Args:
            new_mode: geodesic_path indicates using the simulator's shortest
                path algorithm to find points on the map to navigate between.
                greedy indicates trying to move forward at all possible
                orientations and selecting the one which reduces the geodesic
                distance the most.
        """
        assert new_mode in {"geodesic_path", "greedy"}
        if new_mode == "geodesic_path":
            assert (
                getattr(self._sim, "get_straight_shortest_path_points", None)
                is not None
            )
        self._mode = new_mode
    
    def reset_metric(self, episode):
        if episode.goals[0].radius is not None:
            self.goal_radius = episode.goals[0].radius
        self._return_one_hot = False
        self._max_delta = self._sim.config.FORWARD_STEP_SIZE - 1e-6
        self._step_size = self._sim.config.FORWARD_STEP_SIZE

        self._mode = (
            "geodesic_path"
            if getattr(self._sim, "get_straight_shortest_path_points", None)
            is not None
            else "greedy"
        )
        self._return_one_hot = False
        self._metric = 2 #Default start action is turn left
    
    def update_metric(self, episode, action):
        if action == self._sim.index_stop_action:
            self._metric = 2 #Default start action is turn left
        else:
            self._metric = self.get_next_action(episode.goals[0].position) 

@registry.register_measure
class RoomNavMetric(Measure):
    r"""RoomNavMetric - SPL but for RoomNav
    """
    def __init__(self, sim: Simulator, config: Config):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "roomnavmetric"

    def reset_metric(self, episode):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        # self._start_end_episode_distance = self.nearest_point_in_room(episode.start_position, episode.goals[0].position, episode.goals[0].room_aabb)
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def in_room(self, position, room_aabb):
        assert(room_aabb[1] < room_aabb[3] and room_aabb[0] < room_aabb[2])
        if (
            position[0] > room_aabb[0]-0.1 and position[2] > room_aabb[1]-0.1 
            and position[0] < room_aabb[2]+0.1 and position[2] < room_aabb[3]+0.1
        ):
            return True

        return False

    # def update_metric(self, episode, action):
    #     ep_success = 0
    #     current_position = self._sim.get_agent_state().position.tolist()

    #     distance_to_target = self._sim.geodesic_distance(
    #         current_position, episode.goals[0].position
    #     )

    #     if (
    #         action == self._sim.index_stop_action
    #         and distance_to_target < 0.5
    #     ):
    #         ep_success = 1

    #     self._agent_episode_distance += self._euclidean_distance(
    #         current_position, self._previous_position
    #     )

    #     self._previous_position = current_position

    #     self._metric = ep_success * (
    #         self._start_end_episode_distance
    #         / max(
    #             self._start_end_episode_distance, self._agent_episode_distance
    #         )
    #     )

    def update_metric(self, episode, action):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        # print("ROOM GOAL AABB:", episode.goals[0].room_aabb)
        if (
            action == self._sim.index_stop_action
            and self.in_room(current_position, episode.goals[0].room_aabb)
        ):
            ep_success = 1
            
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collisions"

    def reset_metric(self, episode):
        self._metric = None

    def update_metric(self, episode, action):
        if self._metric is None:
            self._metric = 0

        current_position = self._sim.get_agent_state().position
        if (
            action == self._sim.index_forward_action
            and self._sim.distance_to_closest_obstacle(current_position)
            < COLLISION_PROXIMITY_TOLERANCE
        ):
            self._metric += 1


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self, episode):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._config.DRAW_SOURCE_AND_TARGET:
            # mark source point
            s_x, s_y = maps.to_grid(
                episode.start_position[0],
                episode.start_position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )

            point_padding = 2 * int(
                np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
            )
            top_down_map[
                s_x - point_padding : s_x + point_padding + 1,
                s_y - point_padding : s_y + point_padding + 1,
            ] = maps.MAP_SOURCE_POINT_INDICATOR

            # mark target point
            t_x, t_y = maps.to_grid(
                episode.goals[0].position[0],
                episode.goals[0].position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )

            top_down_map[
                t_x - point_padding : t_x + point_padding + 1,
                t_y - point_padding : t_y + point_padding + 1,
            ] = maps.MAP_TARGET_POINT_INDICATOR

        return top_down_map

    def reset_metric(self, episode):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map(episode)
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

    def update_metric(self, episode, action):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        house_map = house_map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

        self._metric = {
            "map": house_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
        }

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = int(
                np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            )
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y


@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self,
        task_config: Config,
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:

        task_measurements = []
        for measurement_name in task_config.MEASUREMENTS:
            measurement_cfg = getattr(task_config, measurement_name)
            measure_type = registry.get_measure(measurement_cfg.TYPE)
            assert (
                measure_type is not None
            ), "invalid measurement type {}".format(measurement_cfg.TYPE)
            task_measurements.append(measure_type(sim, measurement_cfg))
        self.measurements = Measurements(task_measurements)

        task_sensors = []
        for sensor_name in task_config.SENSORS:
            sensor_cfg = getattr(task_config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)
            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            task_sensors.append(sensor_type(sim, sensor_cfg))

        self.sensor_suite = SensorSuite(task_sensors)
        super().__init__(config=task_config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)
