

import shutil
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


# Function to link assets from the library
def load_assets(library_path, assets, link=False):
    # Open the library and import the specified assets
    with bpy.data.libraries.load(library_path, link=link) as (data_from, data_to):
        for category, asset_names in assets.items():
            # Check if the category exists in the library
            if hasattr(data_from, category):
                available_assets = getattr(data_from, category)
                print(f"Category: {category}")
                print(f"  Available Assets: {available_assets}")
                
                # Find and load the specified assets
                assets_to_load = [name for name in asset_names if name in available_assets]
                if assets_to_load:
                    setattr(data_to, category, assets_to_load)
                    print(f"  Imported Assets: {assets_to_load}")
                else:
                    print(f"  No matching assets found for: {asset_names}")
                    

#######################################################################################################################
def apply_geometry_node_group(object, geometry_node_name):
    # Apply the default Geometry Nodes modifier
    if geometry_node_name in bpy.data.node_groups:
        geom_node_group = bpy.data.node_groups[geometry_node_name]
        
        # Add a Geometry Nodes modifier
        geom_modifier = object.modifiers.new(name="GeometryNodes", type='NODES')
        geom_modifier.node_group = geom_node_group
        print(f"Applied Geometry Node group '{geometry_node_name}' to {object.name}")
    else:
        print(f"Geometry Node group '{geometry_node_name}' not found.")
    
            

def create_blender_scene():     
    
    library_path = "output/template.blend"
    
    
    # Names of the assets in the library
    default_geometry_node = "Output Geometry"  # Name of the Geometry Nodes group
    default_material = "Output Material"       # Name of the Material    
    # assets = {"materials": ['Output Material', 'Input Material'],
    #           "node_groups": ['Output Geometry', 'Input Geometry'],
    #           "worlds": ['World']}
    
    frame_collection_name = 'Collection'

    
    cameras = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    lasers = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
    # lasers = ['TOP']
    
    # debug
    # cameras = []
    # lasers = []

    
    input_dp = 'data/waymo/segment-10584247114982259878_490_000_510_000_with_camera_labels'
    # input_dp = 'data/waymo/segment-10455472356147194054_1560_000_1580_000_with_camera_labels'
    input_dp = 'data/waymo/segment-10837554759555844344_6525_000_6545_000_with_camera_labels'
    output_dp = 'output'

    # in m
    sensor_width = 0.036

    # in m
    image_plane_distance = 100
    combine_lasers = True
    
    workspace = Path.cwd()
    output_dp = Path(output_dp)
    output_image_dp = output_dp / 'images'
    output_image_dp.mkdir(parents=True, exist_ok=True)

    image_dp = Path(input_dp) / 'images'
    point_cloud_dp = Path(input_dp) / 'pointcloud'
    calibration_dp = Path(input_dp) / 'calibration'
    
    frame_id = 55

    image_dimensions = []
    max_image_width, max_image_height = 0, 0
    for camera_name in cameras:
        image_fp = image_dp / camera_name / f"{frame_id:05d}_{camera_name}.jpg"
        image = cv2.imread(str(image_fp))
        image_dimensions.append(image.shape)
        max_image_width = max(max_image_width, image.shape[1])
        max_image_height = max(max_image_height, image.shape[0])
        print('Image resolution:', image.shape)
    
    # Create a new Blender file
    bpy.ops.wm.read_factory_settings()       
    
    # Open the template blend file
    bpy.ops.wm.open_mainfile(filepath=library_path)
    
    # load_assets(library_path, {'scenes': ['Scene']})
    
    # scene_name = bpy.data.scenes[0].name
    # bpy.data.scenes.remove(bpy.data.scenes[scene_name])
    # bpy.data.scenes[-1].name = scene_name   
    
    collection = bpy.data.collections.get(frame_collection_name)
    if collection:
        # Remove all objects in the collection
        for obj in list(collection.objects):  # Use list() to avoid modification during iteration
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"Cleared all objects in the collection: {frame_collection_name}")
    else:
        print(f"Collection '{collection_name}' not found.")
        
    
    pcd_fps = [point_cloud_dp / laser_name / f"{frame_id:05d}_{laser_name}.ply" 
               for laser_name in lasers]
        
    if combine_lasers and len(pcd_fps) > 1:
        pcs = [o3d.io.read_point_cloud(str(pcd_fp)) for pcd_fp in pcd_fps]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(pc.points) for pc in pcs]))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack([np.asarray(pc.colors) for pc in pcs]))
        comnbined_pcd_fp = output_dp / f"{frame_id:05d}_COMBINED.ply"
        o3d.io.write_point_cloud(str(comnbined_pcd_fp), pcd)
        bpy.ops.wm.ply_import(filepath=str(comnbined_pcd_fp))
        imported_object = bpy.context.view_layer.objects.active
        apply_geometry_node_group(imported_object, default_geometry_node)
    else:
        for pcd_fp in pcd_fps:
            bpy.ops.wm.ply_import(filepath=str(pcd_fp))
            imported_object = bpy.context.view_layer.objects.active
            apply_geometry_node_group(imported_object, default_geometry_node)
        
    for img_dim, camera_name in zip(image_dimensions, cameras):
        
        # load camera calibration
        intrinsics_fp = calibration_dp / f"intrinsics_{camera_name}.npz"
        intrinsics = np.load(intrinsics_fp, allow_pickle=True)        
        f_u, f_v, c_u, c_v = intrinsics['f_u'], intrinsics['f_v'], intrinsics['c_u'], intrinsics['c_v']
        
        # Intrinsic matrix (3x3)
        K = np.array([[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]])
        
        # Distortion coefficients
        dist_coeffs = np.array(intrinsics)

        fov_u = 2 * np.arctan(max_image_width / (2 * f_u))
        fov_v = 2 * np.arctan(max_image_height / (2 * f_v))
        
        # focal length in m
        focal_length = f_u / max_image_width * sensor_width
        
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

        # T_vehicle2cam = np.vstack((np.hstack((R.T, -R.T @ t)), np.array([0, 0, 0, 1])))
        
        # create image plane
        img_aspect_ratio = img_dim[1] / img_dim[0]

        # image plane dimensions per default are: aspect ratio x 1.0
        plane_height = 1.0
        plane_width = img_aspect_ratio * plane_height
        plane_scale = sensor_width / focal_length * image_plane_distance / plane_width

        # Calculate offsets based on principal point
        # TODO check if this is correct
        offset_x = -(c_u - (img_dim[1] / 2)) * plane_width / img_dim[1]
        offset_y = (c_v - (img_dim[0] / 2)) * plane_height / img_dim[0]

        scale_matrix = Matrix.Scale(plane_scale, 4)
        plane_distance_matrix = Matrix.Translation(Vector((0, 0, -image_plane_distance)))

        # Create a translation matrix for the offset
        principle_offset_matrix = Matrix.Translation(Vector((offset_x, offset_y, 0)))

        # Combine the offset with the original camera-to-vehicle transformation
        T_cam2vehicle_with_offset = T_cam2vehicle @ plane_distance_matrix @ scale_matrix @ principle_offset_matrix

        image_fp = image_dp / camera_name / f"{frame_id:05d}_{camera_name}.jpg"       
        shutil.copy(image_fp, output_image_dp / image_fp.name)
        bpy.ops.image.import_as_mesh_planes(shader='SHADELESS', files=[{'name':str(output_image_dp / image_fp.name)}])
        image_mesh = bpy.context.object
        image_mesh.name = f'{camera_name}_IMAGE'
        image_mesh.matrix_world = T_cam2vehicle_with_offset
               
        camera = bpy.data.cameras.new(name=camera_name)
        camera_object = bpy.data.objects.new(camera_name, camera)
        bpy.context.collection.objects.link(camera_object)

        # Position the camera
        # camera_object.location = tuple(t)  # Adjust as needed
        camera.lens = focal_length * 1000  # Convert to mm as required by Blender
        camera_object.matrix_world = T_cam2vehicle  # Adjust as needed

        # Set the camera as the active camera
        if camera_name == 'FRONT':
            bpy.context.scene.camera = camera_object
        else:
            # set inactive for rendering and visibility to false
            image_mesh.hide_render = True
            image_mesh.hide_viewport = True  
            
            
    # Set the resolution
    bpy.context.scene.render.resolution_x = max_image_width
    bpy.context.scene.render.resolution_y = max_image_height
        
    output_path = output_dp / "highway_output_file.blend"  # Update with your desired output path
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        
    print('Done')
           
                


#######################################################################################################################   
def main():
    create_blender_scene()
    

#######################################################################################################################
if __name__ == '__main__':
    main()

