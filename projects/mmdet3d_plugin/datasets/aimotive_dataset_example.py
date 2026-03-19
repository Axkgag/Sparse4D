from typing import List, Dict, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, SequentialSampler

from src.aimotive_dataset import AiMotiveDataset
from src.data_loader import DataItem
from src.render_functions import get_2d_bbox_of_3d_bbox
from src.loaders.camera_loader import CameraData
from src.traffic_light import trafficlightcolor2int, get_lightcolors
from src.traffic_sign import signType2int

to_tensor = T.ToTensor()

class AiMotiveTorchDataset(Dataset):
    """
    PyTorch Dataset for loading data to PyTorch framework.
    """
    def __init__(self, root_dir: str, max_objects: int = 30, object_type: str = "traffic_light",
                 get_2d_bboxes: bool = False):
        self.dataset = AiMotiveDataset(root_dir, object_type)
        self.object_type = object_type
        self.max_objects = max_objects
        self.get_2d_bboxes = get_2d_bboxes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset.data_loader[self.dataset.dataset_index[index]]
        sensor_data = self.get_sensor_data(data_item)
        annotations = self.get_targets(data_item.annotations.objects, data_item.camera_data)
        annotations = self.prepare_annotations(annotations)

        return sensor_data, annotations

    def get_sensor_data(self, data_item: DataItem) -> List:
        camera_data = data_item.camera_data

        front_wide_cam, front_narrow_cam, left_cam, right_cam = self.prepare_camera_data(camera_data)

        sensor_data = [[front_wide_cam, front_narrow_cam, left_cam, right_cam]]

        return sensor_data

    def get_targets(self, annotations: List[Dict], camera_data):
        # Generate your custom target representation here.
        targets = []
        for obj in annotations:
            x, y, z = [obj[f'BoundingBox3D Origin {ax}'] for ax in ['X', 'Y', 'Z']]
            l, w, h = [obj[f'BoundingBox3D Extent {ax}'] for ax in ['X', 'Y', 'Z']]
            q_w, q_x, q_y, q_z = [obj[f'BoundingBox3D Orientation Quat {ax}'] for ax in ['W', 'X', 'Y', 'Z']]
            if self.object_type == "traffic_sign":
                cat = signType2int[obj['ObjectMeta']['SubType']]
            else:
                color = get_lightcolors(obj)
                cat = trafficlightcolor2int[color]
            targets_ = [cat, x, y, z, l, w, h, q_w, q_x, q_y, q_z]
            if self.get_2d_bboxes:
                cams = [camera_data.front_wide_camera.camera_params, camera_data.front_narrow_camera.camera_params,
                        camera_data.left_camera.camera_params, camera_data.right_camera.camera_params]
                for cam in cams:
                    bbox = get_2d_bbox_of_3d_bbox(obj, cam)
                    targets_.extend(bbox)
            targets.append(torch.tensor([targets_]))

        return targets

    def prepare_camera_data(self, camera_data: CameraData) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        front_wide_cam = to_tensor(camera_data.front_wide_camera.image)
        front_narrow_cam = to_tensor(camera_data.front_narrow_camera.image)
        left_cam = to_tensor(camera_data.left_camera.image)
        right_cam = to_tensor(camera_data.right_camera.image)

        return front_wide_cam, front_narrow_cam, left_cam, right_cam

    def pad_data(self, data: torch.tensor, max_items: int, attributes: int) -> torch.tensor:
        if len(data) > max_items:
            data = data[:max_items]
            return data
        else:
            padded_data = torch.zeros([max_items, attributes])
            padded_data[:data.shape[0], :] = data

        return padded_data

    def prepare_annotations(self, annotations: torch.tensor) -> torch.tensor:
        if len(annotations) > self.max_objects:
            annotations = annotations[:self.max_objects]
        else:
            pad = self.max_objects - len(annotations)
            num_of_features = annotations[0].shape[1]
            for i in range(pad):
                annotations.append(torch.zeros(num_of_features))

        annotations = torch.vstack(annotations)

        return annotations


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    root_directory = 'data'
    train_dataset = AiMotiveTorchDataset(root_directory, max_objects=5, object_type="traffic_light", get_2d_bboxes=True)
    train_sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=8,
    )

    def to_device(d, dev):
        if isinstance(d, (list, tuple)):
            return [to_device(x, dev) for x in d]
        else:
            return d.to(dev)

    step = 0
    for data in train_loader:
        step += 1
        sensor_data, annotation = data
        sensor_data, annotation = to_device(sensor_data, device), to_device(annotation, device)
        print('Iter ', step, annotation[0][0])
