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

def is_valid_target(t, regions, level_y_bounds):

    source_regions = {}
    target_room_id = -1
    target_room_bb = -1
    target_room_type = ''

    flag = False

    #level check   

    if t[1] > level_y_bounds[0] and t[1] < level_y_bounds[1]:
        for region_id, val in regions.items():
            if t[0] > val['box'][0] and t[2] > val['box'][1] and t[0] < val['box'][2] and t[2] < val['box'][3]:
            # if np.power(np.power(np.array(t) - np.array(val['center']), 2).sum(0), 0.5) <= 0.5:
                target_room_id = region_id
                target_room_bb = val['box']
                target_room_type = val['type']
                flag = True

        if flag == True:
            for region_id, val in regions.items():
                if val['type'] != target_room_type:
                    source_regions[val['box']] = val['type']

            return True, source_regions, target_room_id, target_room_bb, target_room_type
    
    return False, source_regions, target_room_id, target_room_bb, target_room_type

def is_compatible_episode(s, t, target_room_aabb, sim, source_regions, level_y_bounds):

    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, '', 0

    d_separation = sim.geodesic_distance(s, t)

    if d_separation == np.inf or d_separation <= 1:
        return False, '', 0

    #overlap check
    if (
        s[0] > target_room_aabb[0] and s[2] > target_room_aabb[1]
        and s[0] < target_room_aabb[2] and s[2] < target_room_aabb[3]
    ):
        return False, '', 0

    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, '', d_separation

    #source level check
    if s[1] > level_y_bounds[0] and s[1] < level_y_bounds[1]:
        for region, region_type in source_regions.items():
            # print(region)
            if s[0] > region[0] and s[2] > region[1] and s[0] < region[2] and s[2] < region[3]:
                return True, region_type, d_separation

    return False, '', 0

def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    start_room_type,
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
        start_room=start_room_type,
        shortest_paths=shortest_paths,
        info=info,
    )

def generate_roomnav_episode(sim, episode_id, regions, level_y_bounds, is_gen_shortest_path = False, shortest_path_success_distance = 0.2,
                                shortest_path_max_steps = 500, closest_dist_limit = 1, furthest_dist_limit = 30, geodesic_to_euclid_min_ratio = 1.1,
                                number_retries_per_target = 1000):
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

        valid_target, source_regions, target_room_id, target_room_aabb, target_room_type = is_valid_target(target_position, regions, level_y_bounds)

        if valid_target == False:
            continue

        # print('TARGET:',target_position)
        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            # try:
            # print(source_position,
            #     target_position,
            #     sim,source_regions, level_y_bounds)

            # print(source_position)
            # import pdb
            # pdb.set_trace()

            is_compatible, source_room_type, dist = is_compatible_episode(
                source_position,
                target_position,
                target_room_aabb,
                sim,
                source_regions,
                level_y_bounds
                )
            # except:
            #     import pdb; pdb.set_trace()

            if is_compatible:
                # print('SOURCE:', source_position)
                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                shortest_paths = None
                if is_gen_shortest_path:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]

                episode = _create_episode(
                    episode_id=episode_id,
                    scene_id=sim.config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    start_room_type=source_room_type,
                    target_position=target_position,
                    target_room_aabb=target_room_aabb,
                    target_room_type=target_room_type,
                    shortest_paths=shortest_paths,
                    radius=shortest_path_success_distance,
                    info={"geodesic_distance": dist},
                )
                return episode

def get_mp3d_scenes(split: str = "train", scene_template: str = "{scene}") -> List[str]:
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
    parser.add_argument('--count-points', default=1000, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--output-path',
                        default="/private/home/medhini/navigation-analysis-habitat/RoomNavHabitat/data/datasets/roomnav/mp3d/v1/test/test",
                        type=str)
    parser.add_argument('-g', '--gpu', default=0, type=int)
    parser.add_argument('--scene-path', type=str,
                        default="/private/home/medhini/navigation-analysis-habitat/RoomNavHabitat/data/scene_datasets/mp3d/{scene}/{scene}.glb")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # scenes = util.get_mp3d_scenes(args.split, args.scene_path)
    scenes = get_mp3d_scenes(args.split, args.scene_path)
    progress_bar = tqdm(total=args.count_points)
    dataset = RoomNavDatasetV1()
    # dataset = PointNavDatasetV1()
    np.random.seed(args.seed)
    difficulty_counts = {}
    allowed_regions = ['bedroom', 'bathroom', 'kitchen', 'living room', 'dining room']


    scene_count = 0

    config = habitat.get_config(config_paths='tasks/pointnav_mp3d.yaml')
    config.defrost()
    config.DATASET.POINTNAVV1.DATA_PATH = '/private/home/medhini/mp3d_dummy/test/test.json.gz'
    config.DATASET.SCENES_DIR = '/private/home/medhini/navigation-analysis-habitat/RoomNavHabitat/data/scene_datasets/'
    config.freeze()

    env = habitat.Env(config=config)

    # roomnav_scenes = {'val':4, 'train':32, 'test':10}

    for i in range(len(env.episodes)):
        obs = env.reset()
        # print('SCENE PATH used:', episode['scene_id'])
        scene = env.sim.semantic_annotations()
        regions = {}

        print('SCENE PATH used:', env.sim.config.SCENE)

        if env.sim.config.SCENE.split('/')[-1] in ['8194nk5LbLH.glb', 'B6ByNegPMKs.glb', 'ZMojNkEp431.glb']:
            print('FAULTY SCENE PATH:', env.sim.config.SCENE)
            continue

        this_scene = False
        level = scene.levels[0]
        level_y_bounds = [level.aabb.center[1] - level.aabb.sizes[1]/2, level.aabb.center[1] + level.aabb.sizes[1]/2] 
        
        for region in level.regions:
            if region.category.name() == 'kitchen':
                scene_count += 1
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
            # for episode_id in range(round(args.count_points / roomnav_scenes[args.split])):
            for episode_id in range(round(args.count_points / len(scenes))):
                # print('EPISODE ID: ', episode_id)
                episode = generate_roomnav_episode(env.sim, episode_id, regions, level_y_bounds)
                # if episode is not None:
                dataset.episodes.append(episode)
                progress_bar.update(1)

    env.close()
    del env

    print('Number of scenes:', scene_count)
    json_str = str(dataset.to_json())

    with gzip.GzipFile(args.output_path + ".json.gz", 'wb') as f:
        f.write(json_str.encode("utf-8"))

    # print("simple_episodes : ", simple_episodes)
    # print("difficulty_counts: ", difficulty_counts)
    # print("Retries per episode: ", RETRIES / len(dataset.episodes))
    # print("RETRIES_DIFF_LEVELS per episode: ",
    #       RETRIES_DIFF_LEVELS / len(dataset.episodes))
    # print("RETRIES_NO_PATH per episode: ",
    #       RETRIES_NO_PATH / len(dataset.episodes))
    # print("RETRIES_DIS_RANGE per episode: ",
    #       RETRIES_DIS_RANGE / len(dataset.episodes))
    # print("RETRIES_SAMPL per episode: ",
    #       RETRIES_SAMPL / len(dataset.episodes))

    print("Dataset file: {}.json.gz includes {} scenes.".format(
        args.output_path, scene_count))


if __name__ == '__main__':
    main()