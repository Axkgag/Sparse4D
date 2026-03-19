import math
import os
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
import pyquaternion
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

@DATASETS.register_module()
class AiMotiveTLTSDataset(Dataset):
    """
    AiMotive traffic-light / traffic-sign dataset for Sparse4D.

    Expected directory layout:
      ODD/<sequence>/sensor/camera/<CAM_NAME>/*.jpg
      ODD/<sequence>/sensor/calibration/*.json
      ODD/<sequence>/sensor/gnssins/egomotion2.json
      ODD/<sequence>/<traffic_light|traffic_sign>/box/3d_body/frame_xxxxxxx.json
    """

    DEFAULT_CAM_ORDER = [
        "F_CTCAM_L",
        "F_CTCAM_R",
        "F_LONGRANGECAM_C",
        "F_MIDRANGECAM_C",
    ]

    def __init__(
        self,
        data_root: str,
        pipeline: Optional[List[Dict]] = None,
        object_type: str = "traffic_light",
        classes: Optional[List[str]] = None,
        cam_order: Optional[List[str]] = None,
        load_interval: int = 1,
        test_mode: bool = False,
        data_aug_conf: Optional[Dict] = None,
        with_seq_flag: bool = False,
        sequences_split_num: int = 1,
        keep_consistent_seq_aug: bool = True,
    ):
        super().__init__()
        assert object_type in [
            "traffic_light",
            "traffic_sign",
        ], f"Unsupported object_type={object_type}"
        self.data_root = data_root
        self.object_type = object_type
        self.test_mode = test_mode
        self.load_interval = load_interval
        self.data_aug_conf = data_aug_conf
        self.with_seq_flag = with_seq_flag
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None

        self.cam_order = cam_order if cam_order is not None else self.DEFAULT_CAM_ORDER

        self.CLASSES = tuple(classes) if classes is not None else self._default_classes(object_type)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.data_infos = self._build_infos()

        if self.with_seq_flag:
            self._set_sequence_group_flag()

    @staticmethod
    def _default_classes(object_type: str) -> Tuple[str, ...]:
        if object_type == "traffic_light":
            return ("red", "yellow", "green", "off", "unknown")
        return (
            "speed_limit",
            "yield",
            "stop",
            "no_entry",
            "priority",
            "unknown",
        )

    def __len__(self) -> int:
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        flags = []
        curr = 0
        prev_seq = None
        for info in self.data_infos:
            seq = info["sequence_id"]
            if prev_seq is not None and seq != prev_seq:
                curr += 1
            flags.append(curr)
            prev_seq = seq
        self.flag = np.array(flags, dtype=np.int64)

    def get_cat_ids(self, idx: int) -> List[int]:
        info = self.data_infos[idx]
        gt_names = set(info.get("gt_names", []))
        return [self.cat2id[name] for name in gt_names if name in self.cat2id]

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf.get("rand_flip", False) and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        return {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        if self.pipeline is not None:
            data = self.pipeline(data)
        return data

    def get_data_info(self, index: int) -> Dict:
        info = self.data_infos[index]
        out = {
            "sample_idx": info["sample_idx"],
            "timestamp": info["timestamp"],
            "img_filename": info["img_filename"],
            "lidar2img": [x.copy() for x in info["lidar2img"]],
            "cam_intrinsic": [x.copy() for x in info["cam_intrinsic"]],
            "lidar2global": info["lidar2global"].copy(),
            "lidar2ego_translation": [0.0, 0.0, 0.0],
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation": info["lidar2global"][:3, 3].tolist(),
            "ego2global_rotation": self._rotmat_to_quat(info["lidar2global"][:3, :3]),
            "sweeps": [],
        }
        if not self.test_mode:
            out.update(self.get_ann_info(index))
        return out

    def get_ann_info(self, index: int) -> Dict:
        info = self.data_infos[index]
        gt_bboxes_3d = info.get("gt_bboxes_3d")
        gt_labels_3d = info.get("gt_labels_3d")
        gt_names = info.get("gt_names")

        if gt_bboxes_3d is None:
            gt_bboxes_3d = np.zeros((0, 9), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
            gt_names = np.array([], dtype=object)

        return {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "gt_names": gt_names,
        }

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        # The official TL/TS benchmark protocol is dataset-specific; keep default
        # behavior lightweight so test-time inference can run without crashing.
        return {}

    def _build_infos(self) -> List[Dict]:
        sequence_dirs = self._find_sequence_dirs(self.data_root)

        infos: List[Dict] = []
        for seq_dir in sequence_dirs:
            infos.extend(self._build_sequence_infos(seq_dir))

        infos = infos[:: self.load_interval]
        return infos

    @staticmethod
    def _find_sequence_dirs(data_root: str) -> List[str]:
        sequence_dirs = []
        for root, dirs, _ in os.walk(data_root):
            if "sensor" in dirs and os.path.isdir(os.path.join(root, "sensor", "camera")):
                sequence_dirs.append(root)
        sequence_dirs = sorted(set(sequence_dirs))
        return sequence_dirs

    def _build_sequence_infos(self, seq_dir: str) -> List[Dict]:
        sequence_id = os.path.basename(seq_dir)
        sensor_dir = os.path.join(seq_dir, "sensor")
        camera_dir = os.path.join(sensor_dir, "camera")
        calib_dir = os.path.join(sensor_dir, "calibration")
        ego_path = os.path.join(sensor_dir, "gnssins", "egomotion2.json")

        ann_dir = os.path.join(
            seq_dir,
            self.object_type,
            "box",
            "3d_body",
        )
        if not os.path.isdir(camera_dir) or not os.path.isdir(ann_dir):
            return []

        calibration = self._load_calibration(calib_dir)
        ego_pose_map = self._load_egomotion(ego_path)
        camera_frames = self._collect_camera_frames(camera_dir)
        ann_frames = self._collect_annotation_frames(ann_dir)

        if self.test_mode:
            if len(ann_frames) > 0:
                common_frames = sorted(set(camera_frames.keys()) & set(ann_frames.keys()))
            else:
                common_frames = sorted(camera_frames.keys())
        else:
            common_frames = sorted(set(camera_frames.keys()) & set(ann_frames.keys()))
        infos: List[Dict] = []
        for frame_idx in common_frames:
            frame_images = camera_frames[frame_idx]
            if any(cam_name not in frame_images for cam_name in self.cam_order):
                continue

            img_filenames = [frame_images[cam_name] for cam_name in self.cam_order]
            cam_calibs = [calibration.get(cam_name, self._identity_cam_calib()) for cam_name in self.cam_order]
            lidar2img, cam_intrinsic = self._build_camera_mats(cam_calibs)

            if frame_idx in ann_frames:
                gt_boxes, gt_labels, gt_names = self._parse_frame_annotations(ann_frames[frame_idx])
            else:
                gt_boxes = np.zeros((0, 9), dtype=np.float32)
                gt_labels = np.zeros((0,), dtype=np.int64)
                gt_names = np.array([], dtype=object)
            lidar2global = ego_pose_map.get(frame_idx, np.eye(4, dtype=np.float32))

            infos.append(
                {
                    "sequence_id": sequence_id,
                    "sample_idx": f"{sequence_id}_{frame_idx:07d}",
                    "timestamp": float(frame_idx),
                    "img_filename": img_filenames,
                    "lidar2img": lidar2img,
                    "cam_intrinsic": cam_intrinsic,
                    "lidar2global": lidar2global,
                    "gt_bboxes_3d": gt_boxes,
                    "gt_labels_3d": gt_labels,
                    "gt_names": gt_names,
                }
            )
        return infos

    @staticmethod
    def _collect_camera_frames(camera_dir: str) -> Dict[int, Dict[str, str]]:
        frames: Dict[int, Dict[str, str]] = {}
        for cam_name in os.listdir(camera_dir):
            cam_path = os.path.join(camera_dir, cam_name)
            if not os.path.isdir(cam_path):
                continue
            for fname in sorted(os.listdir(cam_path)):
                if not (fname.endswith(".jpg") or fname.endswith(".png")):
                    continue
                frame_idx = AiMotiveTLTSDataset._extract_frame_idx(fname)
                if frame_idx is None:
                    continue
                frames.setdefault(frame_idx, {})[cam_name] = os.path.join(cam_path, fname)
        return frames

    @staticmethod
    def _collect_annotation_frames(ann_dir: str) -> Dict[int, str]:
        frames: Dict[int, str] = {}
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            frame_idx = AiMotiveTLTSDataset._extract_frame_idx(fname)
            if frame_idx is None:
                continue
            frames[frame_idx] = os.path.join(ann_dir, fname)
        return frames

    @staticmethod
    def _extract_frame_idx(name: str) -> Optional[int]:
        digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
        if not digits:
            return None
        # Use the last numeric span to match both camera and annotation names.
        return int(digits[-1])

    def _parse_frame_annotations(
        self, frame_json_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = mmcv.load(frame_json_path)
        objects = []
        if isinstance(data, dict):
            if "objects" in data and isinstance(data["objects"], list):
                objects = data["objects"]
            elif "annotations" in data and isinstance(data["annotations"], list):
                objects = data["annotations"]
            else:
                objects = []
        elif isinstance(data, list):
            objects = data

        gt_boxes = []
        gt_labels = []
        gt_names = []
        for obj in objects:
            box = self._obj_to_box(obj)
            if box is None:
                continue
            name = self._obj_to_class_name(obj)
            label = self.cat2id.get(name, -1)
            if label < 0:
                continue
            gt_boxes.append(box)
            gt_labels.append(label)
            gt_names.append(name)

        if len(gt_boxes) == 0:
            return (
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.array([], dtype=object),
            )

        return (
            np.asarray(gt_boxes, dtype=np.float32),
            np.asarray(gt_labels, dtype=np.int64),
            np.asarray(gt_names, dtype=object),
        )

    def _obj_to_class_name(self, obj: Dict) -> str:
        if self.object_type == "traffic_light":
            color = "unknown"
            if isinstance(obj.get("ObjectMeta"), dict):
                color = (
                    obj["ObjectMeta"].get("Color")
                    or obj["ObjectMeta"].get("LightColor")
                    or obj["ObjectMeta"].get("SubType")
                    or "unknown"
                )
            color = str(color).lower()
            if "red" in color:
                return "red"
            if "yellow" in color or "amber" in color:
                return "yellow"
            if "green" in color:
                return "green"
            if "off" in color or "dark" in color:
                return "off"
            return "unknown"

        subtype = "unknown"
        if isinstance(obj.get("ObjectMeta"), dict):
            subtype = obj["ObjectMeta"].get("SubType", "unknown")
        subtype = str(subtype).lower()
        if "speed" in subtype:
            return "speed_limit"
        if "yield" in subtype:
            return "yield"
        if "stop" in subtype:
            return "stop"
        if "entry" in subtype:
            return "no_entry"
        if "priority" in subtype:
            return "priority"
        return "unknown"

    @staticmethod
    def _obj_to_box(obj: Dict) -> Optional[List[float]]:
        try:
            x = float(obj["BoundingBox3D Origin X"])
            y = float(obj["BoundingBox3D Origin Y"])
            z = float(obj["BoundingBox3D Origin Z"])
            l = float(obj["BoundingBox3D Extent X"])
            w = float(obj["BoundingBox3D Extent Y"])
            h = float(obj["BoundingBox3D Extent Z"])
            qw = float(obj["BoundingBox3D Orientation Quat W"])
            qx = float(obj["BoundingBox3D Orientation Quat X"])
            qy = float(obj["BoundingBox3D Orientation Quat Y"])
            qz = float(obj["BoundingBox3D Orientation Quat Z"])
            quat = pyquaternion.Quaternion([qw, qx, qy, qz])
            rot = quat.rotation_matrix
            yaw = math.atan2(rot[1, 0], rot[0, 0])
            vx = float(obj.get("Velocity X", 0.0))
            vy = float(obj.get("Velocity Y", 0.0))
            return [x, y, z, w, l, h, yaw, vx, vy]
        except Exception:
            return None

    @staticmethod
    def _identity_cam_calib() -> Dict:
        return {
            "K": np.eye(3, dtype=np.float32),
            "T_cam_to_ego": np.eye(4, dtype=np.float32),
        }

    def _build_camera_mats(
        self, cam_calibs: List[Dict]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        lidar2img_list = []
        intrinsic_list = []
        for calib in cam_calibs:
            K = np.asarray(calib["K"], dtype=np.float32)
            T_cam_to_ego = np.asarray(calib["T_cam_to_ego"], dtype=np.float32)
            T_ego_to_cam = np.linalg.inv(T_cam_to_ego)

            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:3, :3] = K
            lidar2img = viewpad @ T_ego_to_cam
            lidar2img_list.append(lidar2img.astype(np.float32))

            intrinsic_4x4 = np.eye(4, dtype=np.float32)
            intrinsic_4x4[:3, :3] = K
            intrinsic_list.append(intrinsic_4x4)
        return lidar2img_list, intrinsic_list

    def _load_calibration(self, calib_dir: str) -> Dict[str, Dict]:
        calibration = {}
        calibration_json = os.path.join(calib_dir, "calibration.json")
        extrinsic_json = os.path.join(calib_dir, "extrinsic_matrices.json")

        calib_data = mmcv.load(calibration_json) if os.path.exists(calibration_json) else {}
        extr_data = mmcv.load(extrinsic_json) if os.path.exists(extrinsic_json) else {}

        for cam_name in self.cam_order:
            K = self._parse_intrinsic(calib_data, cam_name)
            T_cam_to_ego = self._parse_extrinsic(extr_data, calib_data, cam_name)
            calibration[cam_name] = {
                "K": K,
                "T_cam_to_ego": T_cam_to_ego,
            }
        return calibration

    @staticmethod
    def _parse_intrinsic(calib_data: Dict, cam_name: str) -> np.ndarray:
        candidates = [
            calib_data.get("camera", {}).get(cam_name, {}),
            calib_data.get(cam_name, {}),
        ]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            for key in ["K", "intrinsic", "camera_matrix", "cam_intrinsic"]:
                if key in item:
                    mat = np.asarray(item[key], dtype=np.float32)
                    if mat.shape == (3, 3):
                        return mat
                    if mat.shape == (4, 4):
                        return mat[:3, :3]
        return np.eye(3, dtype=np.float32)

    @staticmethod
    def _parse_extrinsic(extr_data: Dict, calib_data: Dict, cam_name: str) -> np.ndarray:
        candidates = [
            extr_data.get(cam_name),
            extr_data.get("camera", {}).get(cam_name) if isinstance(extr_data.get("camera"), dict) else None,
            calib_data.get("camera", {}).get(cam_name, {}).get("extrinsic") if isinstance(calib_data.get("camera"), dict) else None,
            calib_data.get(cam_name, {}).get("extrinsic") if isinstance(calib_data.get(cam_name), dict) else None,
        ]
        for c in candidates:
            if c is None:
                continue
            mat = np.asarray(c, dtype=np.float32)
            if mat.shape == (4, 4):
                return mat
            if mat.shape == (3, 4):
                out = np.eye(4, dtype=np.float32)
                out[:3, :] = mat
                return out
        return np.eye(4, dtype=np.float32)

    def _load_egomotion(self, ego_path: str) -> Dict[int, np.ndarray]:
        if not os.path.exists(ego_path):
            return {}
        data = mmcv.load(ego_path)
        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            for key in ["frames", "egomotion", "poses", "data"]:
                if key in data and isinstance(data[key], list):
                    entries = data[key]
                    break

        pose_map = {}
        for entry in entries:
            frame_idx = self._extract_pose_frame_idx(entry)
            if frame_idx is None:
                continue
            pose = self._entry_to_pose(entry)
            pose_map[frame_idx] = pose
        return pose_map

    @staticmethod
    def _extract_pose_frame_idx(entry: Dict) -> Optional[int]:
        for key in ["frame_id", "frame", "index", "idx"]:
            if key in entry:
                try:
                    return int(entry[key])
                except Exception:
                    pass
        for key in ["timestamp", "image_timestamp", "camera_timestamp"]:
            if key in entry:
                try:
                    return int(entry[key])
                except Exception:
                    pass
        return None

    @staticmethod
    def _entry_to_pose(entry: Dict) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        if "matrix" in entry:
            mat = np.asarray(entry["matrix"], dtype=np.float32)
            if mat.shape == (4, 4):
                return mat

        trans = None
        for key in ["translation", "position", "t"]:
            if key in entry and isinstance(entry[key], (list, tuple)) and len(entry[key]) >= 3:
                trans = np.asarray(entry[key][:3], dtype=np.float32)
                break
        if trans is None:
            trans = np.zeros(3, dtype=np.float32)

        quat = None
        for key in ["rotation", "quaternion", "q"]:
            if key in entry and isinstance(entry[key], (list, tuple)) and len(entry[key]) >= 4:
                quat = np.asarray(entry[key][:4], dtype=np.float32)
                break

        if quat is not None:
            q = pyquaternion.Quaternion(quat.tolist())
            pose[:3, :3] = q.rotation_matrix.astype(np.float32)
        pose[:3, 3] = trans
        return pose

    @staticmethod
    def _rotmat_to_quat(rot: np.ndarray) -> List[float]:
        q = pyquaternion.Quaternion(matrix=rot)
        return [q.w, q.x, q.y, q.z]
