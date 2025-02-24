import logging
import pathlib
from tqdm import tqdm
import numpy as np
import cv2
import open3d as o3d
import matplotlib.cm as cm
import pykitti


from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger(__name__)

#######################################################################################################################
class KITTIPreprocessor():
    def __init__(self):
        super().__init__()

    ###################################################################################################################
    def init_sequence(self, sequence_dp, data, cfg):
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
            for camera_name in cfg.dataset.cameras:
                (image_dp / camera_name).mkdir(parents=True, exist_ok=True)
                P = getattr(data.calib, f'P_rect_{camera_name[-1]}0')
                camera_intrinsics = {
                    'f_u': P[0, 0],
                    'f_v': P[1, 1],
                    'c_u': P[0, 2],
                    'c_v': P[1, 2],
                }                        
                # camera to velo/vehicle transformation
                extrinsics = getattr(data.calib, f'T_{camera_name}_velo')
                np.savez(calibration_dp / fnt.intrinsics.format(sensor_type='camera', 
                                                                sensor_name=camera_name), **camera_intrinsics)
                
                R, t = extrinsics[:3, :3], extrinsics[:3, 3]
                c = -R.T @ t
                ax_swap = np.array([[ 0, 0,  1],
                                    [-1, 0,  0],
                                    [ 0, -1,  0]])
                R = ax_swap @ R            
                # t = -R.T @ c
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = c   
                
                np.save(calibration_dp / fnt.extrinsics.format(sensor_type='camera', 
                                                               sensor_name=camera_name), extrinsics)
                
        if cfg.write_point_cloud:
            for laser_name in cfg.dataset.lasers:
                (point_cloud_dp / laser_name).mkdir(parents=True, exist_ok=True)
                extrinsics = np.eye(4)
                np.save(calibration_dp / fnt.extrinsics.format(sensor_type='laser', 
                                                               sensor_name=laser_name), extrinsics)
                
        return image_dp, calibration_dp, point_cloud_dp
    
    
    ###################################################################################################################
    def run(self, cfg : DictConfig) -> None:
        viridis_colormap = cm.get_cmap('viridis')
        dataset_dp = pathlib.Path(cfg.dataset_dir)
        fnt = cfg.dataset.file_name_templates
        
        for sequence in cfg.sequences:
            log.info(f"Processing sequence: {sequence}")

            # Open config for modification and add 'sequence'
            # minor hack, but able to reuse the same config and filename templates
            with open_dict(cfg):
                cfg.sequence = sequence  # now the cache_dirs are set properly
                
            date, drive = sequence.split('/')
            drive = drive.split('_')[-2]
            sequence_dp = dataset_dp / sequence               
            
            data = pykitti.raw(dataset_dp, date, drive, frames=range(cfg.frame_range[0], cfg.frame_range[1], 1))
            image_dp, calibration_dp, point_cloud_dp = self.init_sequence(sequence_dp, data, cfg) 
            
            cam_gens = [getattr(data, f'{c}') for c in cfg.dataset.cameras]
            laser_gens = [getattr(data, f'{l}') for l in cfg.dataset.lasers]
                       
            for frame_idx, (images, pointclouds) in enumerate(zip(zip(*cam_gens), zip(*laser_gens)), start=cfg.frame_range[0]):
                                        
                if cfg.write_images:               
                    for camera_name, image in zip(cfg.dataset.cameras, images):
                        decoded_image = np.array(image)
                        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
                        fp = image_dp / fnt.image.format(sensor_name=camera_name, frame_idx=frame_idx)
                        cv2.imwrite(str(fp), decoded_image)
                
                if cfg.write_point_cloud:
                    for laser_name, pointcloud in zip(cfg.dataset.lasers, pointclouds):
                        # intensity to color
                        colors = viridis_colormap(pointcloud[:, 3])[:, :3]
                        
                        pcd = o3d.geometry.PointCloud()        
                        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])                        
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        fp = point_cloud_dp / fnt.point_cloud.format(sensor_name=laser_name, frame_idx=frame_idx)
                        o3d.io.write_point_cloud(str(fp), pcd)
                    
