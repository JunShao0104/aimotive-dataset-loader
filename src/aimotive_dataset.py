import os

from typing import List

from src.data_loader import DataLoader, DataItem
from src.sequence import Sequence


class AiMotiveDataset:
    """
    Multimodal Autonomous Driving dataset.
    The dataset consists of four cameras, two radars, one lidar sensor, and corresponding
    3D bounding box annotations of dynamic objects.

    Attributes:
        dataset_index: a list of keyframe paths
        data_loader: a DataLoader class for loading multimodal sensor data.
    """
    def __init__(self, root_dir: str, split: str = 'train'):
        """
        Args:
            root_dir: path to the dataset
            split: data split, either train or val
        """
        self.dataset_index = self.get_frames(root_dir, split)
        self.data_loader = DataLoader(self.dataset_index)

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index: int) -> DataItem:
        return self.data_loader[self.dataset_index[index]]

    def get_frames(self, path: str, split: str = 'train') -> List[str]:
        """
        Collects the keyframe paths.

        Args:
            path: path to the dataset
            split: data split, either train or val

        Returns:
            data_paths: a list of keyframe paths

        """
        data_paths = []
        odd_path = os.path.join(path, split)
        for odd in os.listdir(odd_path):
            for seq in os.listdir(os.path.join(odd_path, odd)):
                seq_path = os.path.join(odd_path, odd, seq)
                sequence = Sequence(seq_path)
                data_paths.extend(sequence.get_frames())

        return data_paths


if __name__ == '__main__':
    root_directory = "data"
    train_dataset = AiMotiveDataset(root_directory, split='train')
    print(len(train_dataset)) # 50 DataItem
    for data in train_dataset:
        print(type(data)) # <class 'src.data_loader.DataItem'>
        print(data.annotations.path) # e.g: data/train/nighttime/20210901-194123-00.37.12-00.37.27@Yoda/dynamic/box/3d_body/frame_0033643.json
        print(len(data.annotations.objects)) # 8 objects
        print(type(data.annotations.objects[0])) # dict, each object is a dict
        print(data.lidar_data.top_lidar.point_cloud.shape) # np.array: (N of lidar pts, 5)
        print(data.radar_data.front_radar.point_cloud.shape) # np.array: (N of front radar pts, 5)
        print(data.radar_data.back_radar.point_cloud.shape) # np.array: (N of back radar pts, 5)
        print(data.camera_data.front_camera.image.shape) # np.array: (704, 1280, 3)
        print(data.camera_data.back_camera.image.shape) # np.array: (1216, 1920, 3) attention here!!!
        print(data.camera_data.left_camera.image.shape) # np.array: (704, 1280, 3)
        print(data.camera_data.right_camera.image.shape) # np.array: (704, 1280, 3)
        print('------------------------------------------')
