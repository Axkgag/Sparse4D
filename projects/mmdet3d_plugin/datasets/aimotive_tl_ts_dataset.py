import math
import os
from typing import Any, Dict, List, Optional, Tuple

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
    TL_COLOR_TO_CLASS = {
        ("red", "unknown", "unknown"): "red",
        ("red", "yellow", "unknown"): "red_yellow",
        ("unknown", "yellow", "unknown"): "yellow",
        ("unknown", "unknown", "green"): "green",
        ("unknown", "unknown", "unknown"): "unknown",
    }

    def __init__(
        self,
        data_root: str,
        ann_file: Optional[str] = None,
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
        lazy_init: bool = False,
    ):
        super().__init__()
        assert object_type in [
            "traffic_light",
            "traffic_sign",
        ], f"Unsupported object_type={object_type}"
        self.data_root = data_root
        self.ann_file = ann_file
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
        self.metadata = {}
        if lazy_init:
            self.data_infos = []
        elif self.ann_file is not None:
            self.data_infos = self.load_annotations(self.ann_file)
        else:
            self.data_infos = self._build_infos()

        if self.with_seq_flag:
            self._set_sequence_group_flag()

    @staticmethod
    def _default_classes(object_type: str) -> Tuple[str, ...]:
        if object_type == "traffic_light":
            return ("red", "red_yellow", "yellow", "green", "unknown")
        return ("unknown",)

    def __len__(self) -> int:
        return len(self.data_infos)

    def load_annotations(self, ann_file: str) -> List[Dict]:
        data = mmcv.load(ann_file)
        if isinstance(data, dict) and "infos" in data:
            infos = data["infos"]
            self.metadata = data.get("metadata", {})
        elif isinstance(data, list):
            infos = data
            self.metadata = {}
        else:
            raise ValueError(f"Invalid annotation format in ann_file={ann_file}")

        norm_infos = []
        for info in infos:
            info = dict(info)
            if "img_filename" in info:
                info["img_filename"] = [
                    self._resolve_path(x) for x in info["img_filename"]
                ]

            for key in ["lidar2img", "cam_intrinsic"]:
                if key in info:
                    info[key] = [np.asarray(x, dtype=np.float32) for x in info[key]]

            if "lidar2global" in info:
                info["lidar2global"] = np.asarray(info["lidar2global"], dtype=np.float32)
            else:
                info["lidar2global"] = np.eye(4, dtype=np.float32)

            if "gt_bboxes_3d" in info and info["gt_bboxes_3d"] is not None:
                info["gt_bboxes_3d"] = np.asarray(info["gt_bboxes_3d"], dtype=np.float32)
            if "gt_labels_3d" in info and info["gt_labels_3d"] is not None:
                info["gt_labels_3d"] = np.asarray(info["gt_labels_3d"], dtype=np.int64)
            if "gt_names" in info and info["gt_names"] is not None:
                info["gt_names"] = np.asarray(info["gt_names"], dtype=object)
            if "instance_inds" in info and info["instance_inds"] is not None:
                info["instance_inds"] = np.asarray(
                    info["instance_inds"], dtype=np.int64
                )

            norm_infos.append(info)

        norm_infos = norm_infos[:: self.load_interval]
        return norm_infos

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def _set_sequence_group_flag(self):
        flags = []
        curr = 0
        prev_seq = None
        for info in self.data_infos:
            seq = info.get("sequence_id")
            if seq is None:
                sample_idx = str(info.get("sample_idx", ""))
                seq = sample_idx.rsplit("_", 1)[0]
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
        instance_inds = info.get("instance_inds")

        if gt_bboxes_3d is None:
            gt_bboxes_3d = np.zeros((0, 9), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
            gt_names = np.array([], dtype=object)
            instance_inds = np.zeros((0,), dtype=np.int64)
        elif instance_inds is None:
            # Ensure downstream pipeline always has instance_id in img_metas.
            instance_inds = np.arange(len(gt_labels_3d), dtype=np.int64)
        else:
            instance_inds = np.asarray(instance_inds, dtype=np.int64)

        return {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "gt_names": gt_names,
            "instance_inds": instance_inds,
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
        if metric is None:
            metric = "center_distance"
        if metric not in ["center_distance", "bbox"]:
            raise ValueError(f"Unsupported metric={metric}")

        if not isinstance(results, list):
            raise TypeError("results must be a list")
        if len(results) != len(self):
            raise ValueError(
                f"The length of results is {len(results)} but dataset length is {len(self)}"
            )

        metric_prefix = f"{result_names[0]}_AiMotive"
        match_dist_thr = 2.0
        cls_records = {i: [] for i in range(len(self.CLASSES))}
        num_gts = {i: 0 for i in range(len(self.CLASSES))}

        for idx, result in enumerate(results):
            pred = result.get("img_bbox", result)
            pred_boxes, pred_scores, pred_labels = self._extract_predictions(pred)

            ann = self.get_ann_info(idx)
            gt_boxes = np.asarray(ann["gt_bboxes_3d"], dtype=np.float32)
            gt_labels = np.asarray(ann["gt_labels_3d"], dtype=np.int64)

            for cls_id in range(len(self.CLASSES)):
                gt_mask = gt_labels == cls_id
                gt_cls_boxes = gt_boxes[gt_mask]
                num_gts[cls_id] += len(gt_cls_boxes)

                pred_mask = pred_labels == cls_id
                pred_cls_boxes = pred_boxes[pred_mask]
                pred_cls_scores = pred_scores[pred_mask]

                if len(pred_cls_scores) == 0:
                    continue

                order = np.argsort(-pred_cls_scores)
                pred_cls_boxes = pred_cls_boxes[order]
                pred_cls_scores = pred_cls_scores[order]

                matched = np.zeros(len(gt_cls_boxes), dtype=bool)
                gt_centers = gt_cls_boxes[:, :2] if len(gt_cls_boxes) > 0 else None
                for box, score in zip(pred_cls_boxes, pred_cls_scores):
                    is_tp = 0.0
                    if gt_centers is not None and len(gt_centers) > 0:
                        dists = np.linalg.norm(gt_centers - box[None, :2], axis=1)
                        if len(dists) > 0:
                            match_idx = int(np.argmin(dists))
                            if (
                                dists[match_idx] <= match_dist_thr
                                and not matched[match_idx]
                            ):
                                matched[match_idx] = True
                                is_tp = 1.0
                    cls_records[cls_id].append((float(score), is_tp))

        eval_dict = {}
        aps, precisions, recalls = [], [], []
        for cls_id, cls_name in enumerate(self.CLASSES):
            records = cls_records[cls_id]
            gt_count = num_gts[cls_id]
            ap, precision, recall = self._compute_ap_pr(records, gt_count)
            eval_dict[f"{metric_prefix}/{cls_name}_AP"] = ap
            eval_dict[f"{metric_prefix}/{cls_name}_precision"] = precision
            eval_dict[f"{metric_prefix}/{cls_name}_recall"] = recall
            if gt_count > 0:
                aps.append(ap)
                precisions.append(precision)
                recalls.append(recall)

        eval_dict[f"{metric_prefix}/mAP"] = (
            float(np.mean(aps)) if len(aps) > 0 else 0.0
        )
        eval_dict[f"{metric_prefix}/mPrecision"] = (
            float(np.mean(precisions)) if len(precisions) > 0 else 0.0
        )
        eval_dict[f"{metric_prefix}/mRecall"] = (
            float(np.mean(recalls)) if len(recalls) > 0 else 0.0
        )
        eval_dict[f"{metric_prefix}/match_dist_thr"] = float(match_dist_thr)
        return eval_dict

    @staticmethod
    def _extract_predictions(pred: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = pred.get("boxes_3d", None) if isinstance(pred, dict) else None
        scores = pred.get("scores_3d", None) if isinstance(pred, dict) else None
        labels = pred.get("labels_3d", None) if isinstance(pred, dict) else None

        if boxes is None or scores is None or labels is None:
            return (
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        if hasattr(boxes, "tensor"):
            boxes_np = boxes.tensor.detach().cpu().numpy()
        elif hasattr(boxes, "detach"):
            boxes_np = boxes.detach().cpu().numpy()
        else:
            boxes_np = np.asarray(boxes)

        if hasattr(scores, "detach"):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)

        if hasattr(labels, "detach"):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)

        if boxes_np.size == 0:
            boxes_np = np.zeros((0, 9), dtype=np.float32)
        elif boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]

        return (
            boxes_np.astype(np.float32),
            scores_np.astype(np.float32),
            labels_np.astype(np.int64),
        )

    @staticmethod
    def _compute_ap_pr(
        records: List[Tuple[float, float]], gt_count: int
    ) -> Tuple[float, float, float]:
        if gt_count <= 0:
            return 0.0, 0.0, 0.0
        if len(records) == 0:
            return 0.0, 0.0, 0.0

        records = sorted(records, key=lambda x: x[0], reverse=True)
        tp = np.asarray([x[1] for x in records], dtype=np.float32)
        fp = 1.0 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(float(gt_count), 1.0)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # Precision envelope + integration for robust AP.
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
        return ap, float(precisions[-1]), float(recalls[-1])

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
        sequence_id = os.path.relpath(seq_dir, self.data_root).replace(os.sep, "/")
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
                    "img_filename": [os.path.relpath(x, self.data_root) for x in img_filenames],
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
            if "CapturedObjects" in data and isinstance(data["CapturedObjects"], list):
                objects = data["CapturedObjects"]
            elif "objects" in data and isinstance(data["objects"], list):
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
            if name not in self.cat2id and "unknown" in self.cat2id:
                name = "unknown"
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
            lights = []
            if isinstance(obj.get("ObjectMeta"), dict):
                lights = obj["ObjectMeta"].get("Lights", [])
            if isinstance(lights, list) and len(lights) >= 3:
                key = tuple(str(x.get("color", "unknown")).lower() for x in lights[:3])
                if key in self.TL_COLOR_TO_CLASS:
                    return self.TL_COLOR_TO_CLASS[key]

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
            return "unknown"

        subtype = "unknown"
        if isinstance(obj.get("ObjectMeta"), dict):
            subtype = obj["ObjectMeta"].get("SubType", "unknown")
        subtype = str(subtype).lower()
        if subtype in self.cat2id:
            return subtype

        # Backward-compatible fallback for coarse sign taxonomies.
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
        return subtype

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
            return [x, y, z, l, w, h, yaw, vx, vy]
        except Exception:
            return None

    @staticmethod
    def _identity_cam_calib() -> Dict:
        return {
            "K": np.eye(3, dtype=np.float32),
            "T_ego_to_cam": np.eye(4, dtype=np.float32),
        }

    def _build_camera_mats(
        self, cam_calibs: List[Dict]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        lidar2img_list = []
        intrinsic_list = []
        for calib in cam_calibs:
            K = np.asarray(calib["K"], dtype=np.float32)
            if "T_ego_to_cam" in calib:
                T_ego_to_cam = np.asarray(calib["T_ego_to_cam"], dtype=np.float32)
            else:
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
            T_ego_to_cam = self._parse_extrinsic(extr_data, calib_data, cam_name)
            calibration[cam_name] = {
                "K": K,
                "T_ego_to_cam": T_ego_to_cam,
            }
        return calibration

    @staticmethod
    def _parse_intrinsic(calib_data: Dict, cam_name: str) -> np.ndarray:
        item = calib_data.get(cam_name, {}) if isinstance(calib_data, dict) else {}
        if isinstance(item, dict):
            if "K" in item:
                mat = np.asarray(item["K"], dtype=np.float32)
                if mat.shape == (3, 3):
                    return mat
            if "intrinsic" in item:
                mat = np.asarray(item["intrinsic"], dtype=np.float32)
                if mat.shape == (3, 3):
                    return mat
            if "focal_length_px" in item and "principal_point_px" in item:
                fx, fy = item["focal_length_px"]
                cx, cy = item["principal_point_px"]
                K = np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                return K
        return np.eye(3, dtype=np.float32)

    @staticmethod
    def _parse_extrinsic(extr_data: Dict, calib_data: Dict, cam_name: str) -> np.ndarray:
        key = f"RT_{cam_name}_from_body"
        candidates = [
            extr_data.get(key) if isinstance(extr_data, dict) else None,
            extr_data.get(cam_name) if isinstance(extr_data, dict) else None,
            calib_data.get(cam_name, {}).get("RT_sensor_from_body") if isinstance(calib_data.get(cam_name), dict) else None,
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

        pose_map = {}
        if isinstance(data, dict):
            # Official format: {"206": {"RT_ECEF_body": [[...]] , ...}, ...}
            if all(isinstance(v, dict) for v in data.values()):
                for k, v in data.items():
                    try:
                        frame_idx = int(k)
                    except Exception:
                        frame_idx = self._extract_pose_frame_idx(v)
                    if frame_idx is None:
                        continue
                    pose = self._entry_to_pose(v)
                    pose_map[frame_idx] = pose
                return pose_map

            entries = []
            for key in ["frames", "egomotion", "poses", "data"]:
                if key in data and isinstance(data[key], list):
                    entries = data[key]
                    break
            for entry in entries:
                frame_idx = self._extract_pose_frame_idx(entry)
                if frame_idx is None:
                    continue
                pose = self._entry_to_pose(entry)
                pose_map[frame_idx] = pose
            return pose_map

        if isinstance(data, list):
            for entry in data:
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
        for key in ["timestamp", "image_timestamp", "camera_timestamp", "time", "time_host"]:
            if key in entry:
                try:
                    return int(entry[key])
                except Exception:
                    pass
        return None

    @staticmethod
    def _entry_to_pose(entry: Dict) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        if "RT_ECEF_body" in entry:
            mat = np.asarray(entry["RT_ECEF_body"], dtype=np.float32)
            if mat.shape == (4, 4):
                mat = mat.copy()
                mat[:3, :3] = AiMotiveTLTSDataset._orthonormalize_rotation(mat[:3, :3])
                return mat
        if "matrix" in entry:
            mat = np.asarray(entry["matrix"], dtype=np.float32)
            if mat.shape == (4, 4):
                mat = mat.copy()
                mat[:3, :3] = AiMotiveTLTSDataset._orthonormalize_rotation(mat[:3, :3])
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
    def _orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
        """Project a near-rotation matrix to SO(3) using SVD."""
        rot = np.asarray(rot, dtype=np.float64)
        u, _, vh = np.linalg.svd(rot)
        r = u @ vh
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vh
        return r.astype(np.float32)

    @staticmethod
    def _rotmat_to_quat(rot: np.ndarray) -> List[float]:
        try:
            rot = AiMotiveTLTSDataset._orthonormalize_rotation(rot)
            q = pyquaternion.Quaternion(matrix=rot)
            return [q.w, q.x, q.y, q.z]
        except Exception:
            # Fallback to identity rotation to keep dataloader robust.
            return [1.0, 0.0, 0.0, 0.0]
