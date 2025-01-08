

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

from tools import frustum, utils


#######################################################################################################################
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
        
        
#######################################################################################################################        
def get_image_mesh_plane_resolution(image_mesh):
    material = image_mesh.active_material
    if material and material.node_tree:
        # Find the image texture node
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                image = node.image
                if image:
                    width, height = image.size
                    return width, height
    return 0, 0
    
    
#######################################################################################################################
def create_blender_camera(main_collection, camera_name, intrinsics, extrinsics, image_fp, img_dim, render_dim, 
                          sensor_width, far_plane_scale, near_plane_scale, axis_scaling, frustum_material='Input Material', 
                          adjust_frustum_fov=True):

    f_u, f_v, c_u, c_v = intrinsics['f_u'], intrinsics['f_v'], intrinsics['c_u'], intrinsics['c_v']
        
    # Camera matrix
    # K = np.array([[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]])
    
    # Distortion coefficients
    dist_coeffs = np.array(intrinsics)

    # Calculate the field of view
    # fov_u = 2 * np.arctan(max_image_width / (2 * f_u))
    # fov_v = 2 * np.arctan(max_image_height / (2 * f_v))
    
    max_image_width, max_image_height = render_dim
    
    # Focal length in m
    focal_length = f_u / max_image_width * sensor_width
    image_plane_distance = f_u / max_image_width * far_plane_scale
    
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3:4]
    
    extrinsics = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    extrinsics = Matrix(extrinsics).to_4x4()
            
    # camera to vehicle ax swap: x>-y, y>z, z>-x to be consistent with blender
    ax_swap = np.array([[ 0, 0, -1],
                        [-1, 0,  0],
                        [ 0, 1,  0]])
    R = R @ ax_swap
    T_cam2vehicle = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    T_cam2vehicle = Matrix(T_cam2vehicle).to_4x4()
    
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
    
    # Add image mesh
    bpy.ops.image.import_as_mesh_planes(shader='SHADELESS', files=[{'name':image_fp}])
    image_mesh = bpy.context.object
    bpy.context.collection.objects.unlink(image_mesh)
    main_collection.objects.link(image_mesh)
    image_mesh.name = f'{camera_name}_IMAGE'
    image_mesh.matrix_world = T_cam2vehicle_with_offset    

    # Set the image mesh material
    camera = bpy.data.cameras.new(name=camera_name)        
    camera_obj = bpy.data.objects.new(camera_name, camera)
    main_collection.objects.link(camera_obj)
    # bpy.context.scene.collection.objects.unlink(camera_obj)  

    # Position the camera
    # camera_obj.location = tuple(t)  # Adjust as needed
    camera.lens = focal_length * 1000  # Convert to mm as required by Blender
    camera_obj.matrix_world = T_cam2vehicle  # Adjust as needed
    


    if axis_scaling > 0:
        # Create a new collection for the camera
        frame_name = f'Frame Camera {camera_name}'
        frame_collection = bpy.data.collections.new(frame_name)
        main_collection.children.link(frame_collection)
        
        vehicle_frame = bpy.data.objects.get("Vehicle Frame")
        camera_frame = utils.duplicate_object(vehicle_frame, collection=frame_collection)
        camera_frame.name = frame_name
        camera_frame.matrix_world = extrinsics @ Matrix.Scale(axis_scaling, 4)
        
        frame_collection.hide_viewport = True
        frame_collection.hide_render = True      

    # Set the camera as the active camera
    if camera_name == 'FRONT':
        bpy.context.scene.camera = camera_obj
    else:
        pass
        # set inactive for rendering and visibility to false
        # image_mesh.hide_render = True
        # image_mesh.hide_viewport = True  
        
      
    
    if adjust_frustum_fov:
        vertices = [np.array(image_mesh.matrix_world @ vertex.co) for vertex in image_mesh.data.vertices]       
        vertices = [vertices[idx] for idx in [0, 2, 1, 3]]     # Rearrange the vertices to match the Blender cube vertices
        camera_origin = np.array(camera_obj.matrix_world.translation)
        
        frustum_corners = []
        for i in range(4):
            ray = vertices[i] - camera_origin
            frustum_corners.append(vertices[i])
            frustum_corners.append(camera_origin + ray * near_plane_scale)

        
        # frustum_corners[0] = vertices[0]
        # frustum_corners[2] = vertices[2]
        # frustum_corners[4] = vertices[1]
        # frustum_corners[6] = vertices[3]
        
    else:
        frustum_corners = frustum.get_camera_frustum_corners(camera_obj, far_plane_scale, near_plane_scale)      
        
        
        
            
    
    frustum_obj = frustum.create_frustum(f"Frustum {camera_name}", frustum_corners, thickness=0.01)
    
    # Apply the frustum material
    frustum_obj.data.materials.append(bpy.data.materials[frustum_material])    
    main_collection.objects.link(frustum_obj)
    
       
#######################################################################################################################
def create_blender_scene():     
    
    library_path = "output/template.blend"
    
    
    # Names of the assets in the library
    pc_geometry_node = "Output Geometry"  # Name of the Geometry Nodes group for the point clouds
    pc_material = "Output Material"       # Name of the Material for the point clouds
    frustum_material = "Input Material"   # Name of the Material for the frustums
    
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
    far_plane_scale = 0.55
    near_plane_scale = 0.4
    
    combine_lasers = False
    
    # scaling <0 means no axis
    axis_scaling = 0.2
    
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
        
    render_dim = (max_image_width, max_image_height)
    
    # Create a new Blender file
    bpy.ops.wm.read_factory_settings()       
    
    # Open the template blend file
    bpy.ops.wm.open_mainfile(filepath=library_path)
    
    # load_assets(library_path, {'scenes': ['Scene']})
    
    # scene_name = bpy.data.scenes[0].name
    # bpy.data.scenes.remove(bpy.data.scenes[scene_name])
    # bpy.data.scenes[-1].name = scene_name   
    
    main_collection = bpy.data.collections.get(frame_collection_name)
    if main_collection:
        # Remove all objects in the collection
        for obj in list(main_collection.objects):  # Use list() to avoid modification during iteration
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"Cleared all objects in the collection: {frame_collection_name}")
    else:
        print(f"Collection '{collection_name}' not found.")
        
    pcd_fps = [point_cloud_dp / laser_name / f"{frame_id:05d}_{laser_name}.ply" 
               for laser_name in lasers]
        
    if combine_lasers and len(lasers) > 1:
        pcs = [o3d.io.read_point_cloud(str(point_cloud_dp / laser_name / f"{frame_id:05d}_{laser_name}.ply" )) 
               for pcd_fp in pcd_fps]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(pc.points) for pc in pcs]))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack([np.asarray(pc.colors) for pc in pcs]))
        comnbined_pcd_fp = output_dp / f"{frame_id:05d}_COMBINED.ply"
        o3d.io.write_point_cloud(str(comnbined_pcd_fp), pcd)
        bpy.ops.wm.ply_import(filepath=str(comnbined_pcd_fp))
        imported_obj = bpy.context.view_layer.objects.active
        apply_geometry_node_group(imported_obj, pc_geometry_node)
    else:        
        vehicle_frame = bpy.data.objects.get("Vehicle Frame")
        for laser_name in lasers:
            laser_collection = bpy.data.collections.new(f'Laser {laser_name}')
            main_collection.children.link(laser_collection)
            
            # set lidar visualization
            extrinsics_fp = calibration_dp / f"extrinsics_laser_{laser_name}.npy"
            extrinsics = np.load(extrinsics_fp)
            extrinsics = Matrix(extrinsics).to_4x4()
            
            lidar_obj = bpy.data.objects.get("LiDAR")
            lidar_obj = utils.duplicate_object(lidar_obj, collection=laser_collection)
            lidar_obj.name = f'LiDAR {laser_name}'
            lidar_obj.matrix_world = extrinsics @ Matrix.Scale(0.5, 4)
            lidar_obj.hide_viewport = True
            lidar_obj.hide_render = True
            
            

            
            if axis_scaling > 0:
                 # Create a new collection for the camera
                frame_name = f'Frame Laser {laser_name}'
                frame_collection = bpy.data.collections.new(frame_name)
                laser_collection.children.link(frame_collection)
                
                # the top lidar is not aligned with the vehicle frame as depicted in the paper
                # https://github.com/waymo-research/waymo-open-dataset/issues/726
                laser_frame = utils.duplicate_object(vehicle_frame, collection=frame_collection)
                laser_frame.name = frame_name
                laser_frame.matrix_world = extrinsics @ Matrix.Scale(axis_scaling, 4)     
                
                frame_collection.hide_viewport = True
                frame_collection.hide_render = True           
            
            pcd_fp = point_cloud_dp / laser_name / f"{frame_id:05d}_{laser_name}.ply" 
            bpy.ops.wm.ply_import(filepath=str(pcd_fp))
            imported_obj = bpy.context.view_layer.objects.active
            bpy.context.collection.objects.unlink(imported_obj)
            laser_collection.objects.link(imported_obj)
            apply_geometry_node_group(imported_obj, pc_geometry_node)

    # Create cameras    
    for img_dim, camera_name in zip(image_dimensions, cameras):
        # load camera calibration
        intrinsics_fp = calibration_dp / f"intrinsics_cam_{camera_name}.npz"
        extrinsics_fp = calibration_dp / f"extrinsics_cam_{camera_name}.npy"
        intrinsics = np.load(intrinsics_fp, allow_pickle=True)        
        extrinsics = np.load(extrinsics_fp)
        
        image_fp = image_dp / camera_name / f"{frame_id:05d}_{camera_name}.jpg"       
        shutil.copy(image_fp, output_image_dp / image_fp.name)
        
        camera_collection = bpy.data.collections.new(f'Camera {camera_name}')
        main_collection.children.link(camera_collection)
        
        create_blender_camera(camera_collection, camera_name, intrinsics, extrinsics, 
                              str(output_image_dp / image_fp.name), img_dim, render_dim, 
                              sensor_width, far_plane_scale, near_plane_scale, 
                              axis_scaling, frustum_material)

    # Set the resolution
    bpy.context.scene.render.resolution_x = max_image_width
    bpy.context.scene.render.resolution_y = max_image_height
        
    output_path = output_dp / "output_file.blend"  # Update with your desired output path
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    

        
    print('Done')
           
                
#######################################################################################################################   
def main():
    create_blender_scene()
    

#######################################################################################################################
if __name__ == '__main__':
    main()

