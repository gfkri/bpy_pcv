import logging
import pathlib
from tqdm import tqdm
import numpy as np
import cv2
import open3d as o3d
import matplotlib.cm as cm

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger(__name__)

#######################################################################################################################
class WaymoPreprocessor():
    def __init__(self):
        super().__init__()

    ###################################################################################################################
    def init_sequence(self, frame, cfg):
        """ Initialize folder structure and write information, which is common for all frames in a sequence 
        (camera calibration), to the output folder. """
        
        image_dp = pathlib.Path(cfg.dataset.cache_dirs.images)
        point_cloud_dp = pathlib.Path(cfg.dataset.cache_dirs.point_clouds)
        calibration_dp = pathlib.Path(cfg.dataset.cache_dirs.calibration)
        image_dp.mkdir(parents=True, exist_ok=True)
        point_cloud_dp.mkdir(parents=True, exist_ok=True)
        calibration_dp.mkdir(parents=True, exist_ok=True)
        
        fnt = cfg.dataset.file_name_templates
        
        if cfg.write_images:            
            for calibration in frame.context.camera_calibrations:
                camera_name = open_dataset.CameraName.Name.Name(calibration.name)
                (image_dp / camera_name).mkdir(parents=True, exist_ok=True)
                
                f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = calibration.intrinsic
                camera_intrinsics = {
                    'f_u': f_u,
                    'f_v': f_v,
                    'c_u': c_u,
                    'c_v': c_v,
                    'k1': k1,
                    'k2': k2,
                    'p1': p1,
                    'p2': p2,
                    'k3': k3
                }                        

                extrinsics = np.array(calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
                np.savez(calibration_dp / fnt.intrinsics.format(sensor_type='camera', 
                                                                sensor_name=camera_name), **camera_intrinsics)
                np.save(calibration_dp / fnt.extrinsics.format(sensor_type='camera', 
                                                               sensor_name=camera_name), extrinsics)
                
        if cfg.write_point_cloud:
            for calibration in frame.context.laser_calibrations:
                laser_name = open_dataset.LaserName.Name.Name(calibration.name)
                (point_cloud_dp / laser_name).mkdir(parents=True, exist_ok=True)
                extrinsics = np.reshape(np.array(calibration.extrinsic.transform), [4, 4])
                np.save(calibration_dp / fnt.extrinsics.format(sensor_type='laser', 
                                                               sensor_name=laser_name), extrinsics)
                
        return image_dp, calibration_dp, point_cloud_dp
    
    
    ###################################################################################################################
    def run(self, cfg : DictConfig) -> None:
        viridis_colormap = cm.get_cmap('viridis')
        dataset_dp = pathlib.Path(cfg.dataset_dir)
        fnt = cfg.dataset.file_name_templates

        files = list(dataset_dp.glob('*.tfrecord'))
        files = [f for f in files if f.stem in cfg.sequences]
        
        for fp in tqdm(files):
            log.info(f"Processing file: {fp}")
            sequence = fp.stem
            # Open config for modification and add 'sequence'
            # minor hack, but able to reuse the same config and filename templates
            with open_dict(cfg):
                cfg.sequence = sequence  # now the cache_dirs are set properly
                
            dataset = tf.data.TFRecordDataset(fp, compression_type='')
            for frame_idx, data in enumerate(dataset):
                if frame_idx == 0:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))            
                    image_dp, calibration_dp, point_cloud_dp = self.init_sequence(frame, cfg)
                    
                if cfg.frame_range:
                    if frame_idx < cfg.frame_range[0]:
                        continue
                    elif frame_idx >= cfg.frame_range[1]:
                        break                
                
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                                        
                if cfg.write_images:               
                    for index, camera_image in enumerate(frame.images):
                        camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
                        decoded_image = tf.image.decode_jpeg(camera_image.image).numpy()
                        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
                        fp = image_dp / fnt.image.format(sensor_name=camera_name, frame_idx=frame_idx)
                        cv2.imwrite(str(fp), decoded_image)
                
                if cfg.write_point_cloud:
                    range_images, camera_projections, \
                    _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
                    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, 
                                                                                    camera_projections, 
                                                                                    range_image_top_pose, 
                                                                                    keep_polar_features=True)
                    
                    for laser, point, cp_point in zip(frame.lasers, points, cp_points):
                        # point dimensions are (range, intensity, elongation, x, y, z)
                        rng = point[:, 0]
                        rng_norm = rng / 75.0
                        colors = viridis_colormap(rng_norm)[:, :3]
                        
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(point[:, 3:])                   
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                                            
                        laser_name = open_dataset.LaserName.Name.Name(laser.name)                  
                        
                        fp = point_cloud_dp / fnt.point_cloud.format(sensor_name=laser_name, frame_idx=frame_idx)
                        o3d.io.write_point_cloud(str(fp), pcd)