from typing import Optional
import argparse
import gzip
from typing import List
import csv
import os

import numpy as np
from tqdm import tqdm
import json

import habitat
import numpy as np

from roomnav_dataset import RoomNavDatasetV1

# from pointnav_dataset import PointNavDatasetV1

from habitat.core.simulator import Simulator
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav_task import RoomNavigationEpisode, RoomGoal

"""A minimum radius of a plane that a point should be part of to be
considered  as a target or source location. Used to filter isolated points
that aren't part of a floor.
"""
ISLAND_RADIUS_LIMIT = 1.5

def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2

def nearest_point_in_room(sim, start_position, target_positon, room_aabb):
    x_axes = np.arange(room_aabb[0]+0.2, room_aabb[2]-0.2, 1.0)
    y_axes = np.arange(room_aabb[1]+0.2, room_aabb[3]-0.2, 1.0)

    new_target = None
    
    shortest_distance = 100000.0
    for i in x_axes:
        for j in y_axes:
            if sim.is_navigable([i,target_positon[1],j]):
                dist = sim.geodesic_distance(start_position, [i, target_positon[1], j])
                if dist < shortest_distance:
                    shortest_distance = min(dist, shortest_distance)
                    new_target = [i, target_positon[1], j]

    # assert(new_target!=None)
    if new_target is None:
        new_target = target_positon
        shortest_distance = sim.geodesic_distance(start_position, target_positon)

    return  new_target, shortest_distance

def is_valid_target(t, regions):
    target_room_id = -1
    target_room_bb = -1
    target_room_type = ''

    for region_id, val in regions.items():
        #tigther bounding box room constraints
        if t[0] > val['box'][0]+0.2 and t[2] > val['box'][1]+0.2 and t[0] < val['box'][2]-0.2 and t[2] < val['box'][3]-0.2:
            target_room_id = region_id
            target_room_bb = val['box']
            target_room_type = val['type']
            return True, target_room_id, target_room_bb, target_room_type

    return False, target_room_id, target_room_bb, target_room_type

def is_compatible_episode(sim, s, t, target_room_aabb, closest_dist_limit, furthest_dist_limit, geodesic_to_euclid_min_ratio):

    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0, []
    # t_new, d_separation = t, sim.geodesic_distance(s,t)
    t_new, d_separation = nearest_point_in_room(sim, s, t, target_room_aabb)
    
    # if t_new is None:
    #     return False, 0, []

    if d_separation == np.inf:
        return False, 0, []

    if not closest_dist_limit <= d_separation <= furthest_dist_limit:
        return False, 0, []

    euclid_dist = np.power(np.power(np.array(s) - np.array(t_new), 2).sum(0), 0.5)
    distances_ratio = d_separation / euclid_dist

    if distances_ratio < geodesic_to_euclid_min_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_min_ratio)
    ):
        return False, 0, []

    #overlap check
    if (
        s[0] > target_room_aabb[0] and s[2] > target_room_aabb[1]
        and s[0] < target_room_aabb[2] and s[2] < target_room_aabb[3]
    ):
        return False, 0, []

    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0, []

    #TODO: Another check to see if source is closest to target of target room type.
    
    return True, d_separation, t_new

def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    target_position,
    target_room_type,
    target_room_aabb,
    shortest_paths=None,
    radius=None,
    info=None,
) -> Optional[RoomNavigationEpisode]:
    goals = [RoomGoal(position=target_position, radius=radius, room_aabb=target_room_aabb, room_name=target_room_type)]
    return RoomNavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )

def generate_roomnav_episode(sim, episode_id, regions, is_gen_shortest_path = False, shortest_path_success_distance = 0.2,
                                shortest_path_max_steps = 500, closest_dist_limit = float(4.0), furthest_dist_limit = float(45.0), 
                                geodesic_to_euclid_min_ratio = float(1.2), number_retries_per_target = 1000):
    """Function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    while True:
        target_position = sim.sample_navigable_point()

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        valid_target, target_room_id, target_room_aabb, target_room_type = is_valid_target(target_position, regions)
        
        if valid_target == False:
            continue

        # print('TARGET:',target_position)
        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()

            is_compatible, dist, new_target_position = is_compatible_episode(sim, source_position,target_position,target_room_aabb,
                closest_dist_limit,furthest_dist_limit,geodesic_to_euclid_min_ratio)

            if is_compatible:
                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                shortest_paths = None
                if is_gen_shortest_path:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=new_target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]

                episode = _create_episode(
                    episode_id=episode_id,
                    scene_id=sim.config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    target_position=new_target_position,
                    target_room_aabb=target_room_aabb,
                    target_room_type=target_room_type,
                    shortest_paths=shortest_paths,
                    radius=shortest_path_success_distance,
                    info={"geodesic_distance": dist},
                )

                return episode

def get_mp3d_scenes(split: str = "test", scene_template: str = "{scene}") -> List[str]:
    scenes = []
    with open('scenes_mp3d.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if split in row["set"].split() or split == "*":
                scenes.append(scene_template.format(scene=row["id"]))
    return scenes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=73, type=int)
    parser.add_argument('--count-points', default=2000, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--output-path',
                        default="/private/home/medhini/navigation-analysis-habitat/habitat-api/data/datasets/roomnav/mp3d/v1/test/test_all",
                        type=str)
    parser.add_argument('-g', '--gpu', default=0, type=int)
    parser.add_argument('--scene-path', type=str,
                        default="/private/home/medhini/navigation-analysis-habitat/habitat-api/data/scene_datasets/mp3d/{scene}/{scene}.glb")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    progress_bar = tqdm(total=args.count_points)
    dataset = RoomNavDatasetV1()
    # dataset = PointNavDatasetV1()
    np.random.seed(args.seed)
    allowed_regions = ['bedroom', 'bathroom', 'kitchen', 'living room', 'dining room']

    scene_count = 0

    config = habitat.get_config(config_paths='tasks/pointnav_pointnav_mp3d.yaml')
    config.defrost()
    config.DATASET.DATA_PATH = 'mp3d_dummy/test/test.json.gz'
    config.DATASET.SCENES_DIR = '/private/home/medhini/navigation-analysis-habitat/habitat-api/data/scene_datasets/'
    config.freeze()

    env = habitat.Env(config=config)

    roomnav_scenes = {'val':4, 'train':32, 'test':10}

    for i in range(len(env.episodes)):
        obs = env.reset()
        scene = env.sim.semantic_annotations()
        regions = {}

        print('SCENE PATH used:', env.sim.config.SCENE, env.current_episode.scene_id)

        if env.sim.config.SCENE.split('/')[-1] in ['8194nk5LbLH.glb', 'B6ByNegPMKs.glb', 'ZMojNkEp431.glb']:
            print('FAULTY SCENE PATH:', env.sim.config.SCENE)
            continue

        this_scene = False
        level = scene.levels[0]
        
        for region in level.regions:
            if region.category.name() == 'kitchen':
                this_scene = True
                break

        if this_scene:
            for region in level.regions:
                if region.category.name() in allowed_regions:
                    regions[region.id] = {}
                    regions[region.id]['center'] = region.aabb.center
                    regions[region.id]['box'] = (region.aabb.center[0] - region.aabb.sizes[0]/2, region.aabb.center[2] + region.aabb.sizes[2]/2,
                                                region.aabb.center[0] + region.aabb.sizes[0]/2, region.aabb.center[2] - region.aabb.sizes[2]/2)

                    regions[region.id]['type'] = region.category.name()

            # print(regions)
            if len(regions) > 0:
                scene_count += 1
                for episode_id in range(round(args.count_points / roomnav_scenes[args.split])):
                    episode = generate_roomnav_episode(env.sim, episode_id, regions)
                    dataset.episodes.append(episode)
                    progress_bar.update(1)

    env.close()
    del env

    print('Number of scenes:', scene_count)
    json_str = str(dataset.to_json())

    with gzip.GzipFile(args.output_path + ".json.gz", 'wb') as f:
        f.write(json_str.encode("utf-8"))

    print("Dataset file: {}.json.gz includes {} scenes.".format(
        args.output_path, scene_count))


if __name__ == '__main__':
    main()
