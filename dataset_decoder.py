from pathlib import Path
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



#######################################################################################################################
def write_sequence_information(frame, image_dp, calibration_dp, point_cloud_dp, write_images=True, write_point_cloud=True):
    """ Initialize folder structure and write information, which is common for all frames in a sequence 
    (camera calibration), to the output folder. """
    
    if write_images:
        for index, camera_image in enumerate(frame.images):                    
            camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
            (image_dp / camera_name).mkdir(parents=True, exist_ok=True)
            
            calib = next(cc for cc in frame.context.camera_calibrations
                                if cc.name == camera_image.name)
            f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = calib.intrinsic
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

            extrinsics = np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
            np.savez(calibration_dp / f'intrinsics_{camera_name}.npz', **camera_intrinsics)
            np.save(calibration_dp / f'extrinsics_{camera_name}.npy', extrinsics)
            
    if write_point_cloud:
        for laser in frame.lasers:
            laser_name = open_dataset.LaserName.Name.Name(laser.name)
            (point_cloud_dp / laser_name).mkdir(parents=True, exist_ok=True)


#######################################################################################################################
def decode_waymo_data(write_images=True, write_point_cloud=True):
    
    sequences = {
                 'segment-10584247114982259878_490_000_510_000_with_camera_labels', # night
                 'segment-10455472356147194054_1560_000_1580_000_with_camera_labels', # crossing
                 'segment-10837554759555844344_6525_000_6545_000_with_camera_labels' # highway
                 
                }
    
    frame_id = 55
    viridis_colormap = cm.get_cmap('viridis')
    
    input_dp = '/media/gfkri/data/Datasets/waymo/raw_data/'
    output_dp = '/media/gfkri/data2/Datasets/waymo/evaluation'
    
    input_dp = Path(input_dp)
    output_dp = Path(output_dp)
    
    files = list(input_dp.glob('*.tfrecord'))
    files = [f for f in files if f.stem in sequences]
    
    init = False
    for fp in tqdm(files):
        print(f"Processing file: {fp}")
        
        # Create respective folder in output folder
        segment_dp = output_dp / fp.stem
        
        image_dp = segment_dp / 'images'
        point_cloud_dp = segment_dp / 'pointcloud'
        image_dp.mkdir(parents=True, exist_ok=True)
        point_cloud_dp.mkdir(parents=True, exist_ok=True)
        calibration_dp = segment_dp / 'calibration'
        calibration_dp.mkdir(parents=True, exist_ok=True)

        dataset = tf.data.TFRecordDataset(fp, compression_type='')
        for frame_idx, data in enumerate(dataset):
            if frame_idx == 0:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))            
                write_sequence_information(frame, image_dp, calibration_dp, point_cloud_dp, write_images, write_point_cloud)
                
            if frame_id:
                if frame_idx < frame_id:
                    continue
                elif frame_idx > frame_id:
                    break                
            
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
                                    
            if write_images:               
                for index, camera_image in enumerate(frame.images):
                    camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
                    decoded_image = tf.image.decode_jpeg(camera_image.image).numpy()
                    decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
                    fp = image_dp / camera_name / f"{frame_idx:05d}_{camera_name}.jpg"  
                    cv2.imwrite(str(fp), decoded_image)
            
            if write_point_cloud:
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
                    fp = point_cloud_dp / laser_name / f"{frame_idx:05d}_{laser_name}.ply"
                    o3d.io.write_point_cloud(str(fp), pcd)

                
                
#######################################################################################################################   
def main():
    decode_waymo_data()

    

#######################################################################################################################
if __name__ == '__main__':
    main()