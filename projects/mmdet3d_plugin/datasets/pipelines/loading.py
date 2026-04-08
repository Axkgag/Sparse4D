import json
import os

import cv2
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(
        self,
        to_float32=False,
        color_type="unchanged",
        undistort=False,
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.undistort = undistort
        self._calib_cache = {}
        self._map_cache = {}

    @staticmethod
    def _calibration_json_from_img_path(img_path):
        norm_path = os.path.normpath(img_path)
        marker = f"{os.sep}sensor{os.sep}camera{os.sep}"
        marker_pos = norm_path.find(marker)
        if marker_pos < 0:
            return None, None
        seq_root = norm_path[:marker_pos]
        cam_name = os.path.basename(os.path.dirname(norm_path))
        calib_path = os.path.join(
            seq_root, "sensor", "calibration", "calibration.json"
        )
        return calib_path, cam_name

    def _load_camera_calibration(self, img_path):
        calib_path, cam_name = self._calibration_json_from_img_path(img_path)
        if calib_path is None or cam_name is None or (not os.path.exists(calib_path)):
            return None
        cache_key = (calib_path, cam_name)
        if cache_key in self._calib_cache:
            return self._calib_cache[cache_key]
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        if not isinstance(calib_data, dict):
            self._calib_cache[cache_key] = None
            return None
        cam_cfg = calib_data.get(cam_name)
        if not isinstance(cam_cfg, dict):
            self._calib_cache[cache_key] = None
            return None
        self._calib_cache[cache_key] = cam_cfg
        return cam_cfg

    @staticmethod
    def _get_distortion_coeffs(cam_cfg):
        if not isinstance(cam_cfg, dict):
            return np.zeros((5,), dtype=np.float64)
        coeffs = cam_cfg.get("distortion_coeffs")
        if coeffs is None:
            return np.zeros((5,), dtype=np.float64)
        coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size == 0:
            return np.zeros((5,), dtype=np.float64)
        if coeffs.size < 5:
            coeffs = np.pad(coeffs, (0, 5 - coeffs.size))
        return coeffs[:5]

    @staticmethod
    def _get_source_intrinsic(cam_cfg):
        if not isinstance(cam_cfg, dict):
            return None
        if "focal_length_px" in cam_cfg and "principal_point_px" in cam_cfg:
            fx, fy = cam_cfg["focal_length_px"][:2]
            cx, cy = cam_cfg["principal_point_px"][:2]
            return float(fx), float(fy), float(cx), float(cy)
        if "K" in cam_cfg:
            k = np.asarray(cam_cfg["K"], dtype=np.float64)
            if k.shape == (3, 3):
                return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        return None

    @staticmethod
    def _get_target_intrinsic(results, cam_idx, src_intrinsic):
        cam_intrinsic = results.get("cam_intrinsic")
        if cam_intrinsic is not None and cam_idx < len(cam_intrinsic):
            k = np.asarray(cam_intrinsic[cam_idx], dtype=np.float64)
            if k.shape == (4, 4):
                k = k[:3, :3]
            if k.shape == (3, 3):
                return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        return src_intrinsic

    def _get_undistort_map(
        self,
        model,
        xi,
        dist,
        source_intrinsic,
        target_intrinsic,
        width,
        height,
    ):
        sx, sy, scx, scy = source_intrinsic
        tx, ty, tcx, tcy = target_intrinsic
        key = (
            model,
            float(xi) if xi is not None else None,
            tuple(np.round(dist, 12).tolist()),
            float(sx),
            float(sy),
            float(scx),
            float(scy),
            float(tx),
            float(ty),
            float(tcx),
            float(tcy),
            int(width),
            int(height),
        )
        if key in self._map_cache:
            return self._map_cache[key]

        u, v = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
        )
        x = (u - tcx) / np.clip(tx, 1e-8, 1e8)
        y = (v - tcy) / np.clip(ty, 1e-8, 1e8)

        if model == "mei":
            xi_val = float(xi) if xi is not None else 1.0
            z = np.ones_like(x)
            norm = np.sqrt(x * x + y * y + z * z)
            norm = np.clip(norm, 1e-8, 1e8)
            x = x / norm
            y = y / norm
            z = z / norm + xi_val
            z[np.abs(z) <= 1e-5] = 1e-5
            x = x / z
            y = y / z

        k1, k2, p1, p2, k3 = dist[:5]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        coef = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        qx = x * coef + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        qy = y * coef + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)

        map_x = (qx * sx + scx).astype(np.float32)
        map_y = (qy * sy + scy).astype(np.float32)
        self._map_cache[key] = (map_x, map_y)
        return map_x, map_y

    def _maybe_undistort(self, img, img_path, results, cam_idx):
        if not self.undistort:
            return img
        cam_cfg = self._load_camera_calibration(img_path)
        if cam_cfg is None:
            return img

        model = str(cam_cfg.get("model", "opencv_pinhole")).lower()
        if model not in {"opencv_pinhole", "mei"}:
            return img

        source_intrinsic = self._get_source_intrinsic(cam_cfg)
        if source_intrinsic is None:
            return img
        target_intrinsic = self._get_target_intrinsic(
            results, cam_idx, source_intrinsic
        )
        if target_intrinsic is None:
            return img

        dist = self._get_distortion_coeffs(cam_cfg)
        xi = cam_cfg.get("xi")
        if model == "opencv_pinhole" and np.allclose(dist, 0.0):
            return img

        height, width = img.shape[:2]
        map_x, map_y = self._get_undistort_map(
            model=model,
            xi=xi,
            dist=dist,
            source_intrinsic=source_intrinsic,
            target_intrinsic=target_intrinsic,
            width=width,
            height=height,
        )
        return cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        imgs = []
        for cam_idx, name in enumerate(filename):
            img = mmcv.imread(name, self.color_type)
            img = self._maybe_undistort(img, name, results, cam_idx)
            imgs.append(img)
        # img is of shape (h, w, c, num_views)
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f"undistort={self.undistort})"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        results["points"] = points
        return results
