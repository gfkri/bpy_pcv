

from debug_imports import *
import os
import math
import numpy as np
import itertools
from pathlib import Path
import cv2
from tqdm import tqdm
import open3d as o3d
import bpy
from mathutils import Matrix, Vector

            
                   
def create_blender_scene():     
    cameras = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    lasers = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
    
    input_dp = '/media/gfkri/data2/Datasets/waymo/evaluation/segment-10584247114982259878_490_000_510_000_with_camera_labels'
    output_dp = 'output'
    
    output_dp = Path(output_dp)
    image_dp = Path(input_dp) / 'images'
    point_cloud_dp = Path(input_dp) / 'pointcloud'
    calibration_dp = Path(input_dp) / 'calibration'
    
    frame_id = 0
    
    # Create a new Blender file
    bpy.ops.wm.read_factory_settings()       
    
    # Delete the default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()
    
    for laser_name in lasers:
        pcd_fp = point_cloud_dp / laser_name / f"{frame_id:05d}_{laser_name}.ply"
        bpy.ops.wm.ply_import(filepath=str(pcd_fp))
        
    for camera_name in cameras:
        # load camera calibration

        intrinsics_fp = calibration_dp / f"intrinsics_{camera_name}.npz"
        intrinsics = np.load(intrinsics_fp, allow_pickle=True)        
        f_u, f_v, c_u, c_v = intrinsics['f_u'], intrinsics['f_v'], intrinsics['c_u'], intrinsics['c_v']
        K = np.array([[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]])
        
        extrinsics_fp = calibration_dp / f"extrinsics_{camera_name}.npy"
        T_cam2vehicle = np.load(extrinsics_fp)
        
        R = T_cam2vehicle[:3, :3]
        t = T_cam2vehicle[:3, 3:4]
                
        # camera to vehicle ax swap: x>-y, y>z, z>-x
        ax_swap = np.array([[ 0, 0, -1],
                            [-1, 0,  0],
                            [ 0, 1,  0]])
        R = R @ ax_swap
        T_cam2vehicle = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    
        T_cam2vehicle = Matrix(T_cam2vehicle).to_4x4()

        # T_vehicle2cam = np.vstack((np.hstack((R.T, -R.T @ t)), T_cam2vehicle[3:, :]))
        
        # create image plane
        image_fp = image_dp / camera_name / f"{frame_id:05d}_{camera_name}.jpg"
        bpy.ops.image.import_as_mesh_planes(shader='SHADELESS', files=[{'name':str(image_fp)}])
        image_mesh = bpy.context.object
        image_mesh.matrix_world = T_cam2vehicle
               
        camera = bpy.data.cameras.new(name=camera_name)
        camera_object = bpy.data.objects.new(camera_name, camera)
        bpy.context.collection.objects.link(camera_object)

        # Position the camera
        # camera_object.location = tuple(t)  # Adjust as needed
        camera_object.matrix_world = T_cam2vehicle  # Adjust as needed

        # Set the camera as the active camera
        if camera_name == 'FRONT':
            bpy.context.scene.camera = camera_object
        
    output_path = output_dp / "output_file.blend"  # Update with your desired output path
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        
    print('Done')
           
                


#######################################################################################################################   
def main():
    create_blender_scene()
    plt.show()
    

#######################################################################################################################
if __name__ == '__main__':
    main()

