import os

splits = ['test', 'val', 'train']
root_path = '/private/home/medhini/navigation-analysis-habitat/habitat-api/data/datasets/roomnav/mp3d/v1/'
pointnav_dataset = '/private/home/medhini/navigation-analysis-habitat/habitat-api/data/datasets/pointnav/mp3d/v1/'
pointnav_dataset_v2 = '/private/home/medhini/navigation-analysis-habitat/habitat-api/data/datasets/pointnav/mp3d/v2/'

for split in splits:
    roomnav_scenes = os.listdir(root_path+split+'/content')
    print(len(roomnav_scenes))
    if not os.path.exists(pointnav_dataset_v2+split):
        os.mkdir(pointnav_dataset_v2+split)
        os.mkdir(pointnav_dataset_v2+split+'/content')

    for room in roomnav_scenes:
        pointnav_v1 = os.path.join(pointnav_dataset, split, 'content', room)
        pointnav_v2 = os.path.join(pointnav_dataset_v2, split, 'content')

        os.system('cp %s %s'%(pointnav_v1, pointnav_v2))

    print(len(os.listdir(os.path.join(pointnav_dataset_v2, split, 'content'))))
    os.system('cp %s %s'%(os.path.join(pointnav_dataset, split, split+'.json.gz'), os.path.join(pointnav_dataset_v2, split)))