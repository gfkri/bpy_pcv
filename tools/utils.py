import numpy as np

import bpy


##########################################################################################
def join_objects(objs, name='joined'):
    if len(objs) > 1:
        # fuse all sampled objects to one
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.join()
    else:
        bpy.context.view_layer.objects.active = objs[0]

    merged_object = bpy.context.active_object
    merged_object.name = name
    return merged_object


##########################################################################################
def sample_bounding_box(points, position, dimension, heading):  
    '''Sample all points withing an bounding box'''

    # Create a translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # Create a rotation matrix
    rotation_matrix = np.eye(4)
    rotation_angle = np.radians(heading)
    rotation_matrix[:2, :2] = [[np.cos(heading), -np.sin(heading)],
                              [np.sin(heading), np.cos(heading)]]
                            
    # Create a scaling matrix
    dim_scale_matrix = np.eye(4)
    dim_scale_matrix[:3, :3] = np.diag(dimension)
    
    # Combine the translation, rotation, and scaling matrices
    bounding_box_matrix = translation_matrix @ rotation_matrix @ dim_scale_matrix
    bounding_box_matrix_inv = np.linalg.inv(bounding_box_matrix)

    points_hom = np.concatenate((points, np.ones((len(points), 1))), axis=1)

    # Transform the points into the bounding box's coordinate system
    points_bb = points_hom @ bounding_box_matrix_inv.T
    points_bb = points_bb[:, :3] / points_bb[:, 3:4]
    
    # assuming the origin center lies on the center of the object ground plane
    visible_mask = np.all((np.array((-0.5, -0.5, 0)) <= points_bb) & (points_bb <= np.array((0.5, 0.5, 1.0))), axis=1)
                
    return visible_mask


##########################################################################################
def sample_within_cam_frustum(points, camera):   
    # Get the camera's view matrix
    view_matrix = np.array(camera.matrix_world.inverted())
    
    # Get the camera's projection matrix
    projection_matrix = np.array(camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=bpy.context.scene.render.resolution_x,
        y=bpy.context.scene.render.resolution_y,
        scale_x=bpy.context.scene.render.pixel_aspect_x,
        scale_y=bpy.context.scene.render.pixel_aspect_y
    ))

    # Combine the view and projection matrices
    view_projection_matrix = projection_matrix @ view_matrix
    
    # transform into clip coordinates
    clip_coordinates = transform(points, view_projection_matrix)

    # Check if the coordinates are within the range of -1 to 1
    visible_mask = np.all((-1 <= clip_coordinates) & (clip_coordinates <= 1), axis=1)
            
    return visible_mask


##########################################################################################    
def create_blender_pc(points, name, target_collection=None, replace_if_existing=True):
    if replace_if_existing:      
        original_object = bpy.data.objects.get(name)
    
    mesh = bpy.data.meshes.new(name=name)

    # Create a new object
    obj = bpy.data.objects.new(name, mesh)
    
    collection = bpy.context.collection
    if target_collection is not None:
        collection = bpy.data.collections.get(target_collection)
        if collection is None:
            assert False, f'"{target_collection}" not found!'
    collection.objects.link(obj)
    
    # Construct the mesh
    mesh.from_pydata(points, [], [])
    
    # Update & Free BMesh
    mesh.update()
    
    if replace_if_existing and original_object:      
        # Copy the animation data
        if original_object.animation_data:
            obj.animation_data_create()
            obj.animation_data.action = original_object.animation_data.action.copy()
        
        if original_object.modifiers:
            for original_modifier in original_object.modifiers:
                if original_modifier.type == 'NODES':
                    modifier = obj.modifiers.new(name=original_modifier.name, type='NODES')
                    modifier.node_group = original_modifier.node_group
        
        if original_object.data.materials:             
            obj.data.materials.clear()
            for material in original_object.data.materials:
                obj.data.materials.append(material)        

        bpy.data.objects.remove(original_object)
        obj.name = name
        
    return obj


##########################################################################################
def transform(points, T):
    points_hom = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    points_bb = points_hom @ T.T
    return points_bb[:, :3] / points_bb[:, 3:4]


##########################################################################################
def vertices2array(vertices):
    vlen = len(vertices)
    verts_co_1D = np.zeros([vlen*3], dtype='f')
    vertices.foreach_get("co", verts_co_1D)
    verts_co_3D = verts_co_1D.reshape(vlen, 3)
    return verts_co_3D


##########################################################################################
def duplicate_object(obj, data=True, actions=True, children=True, collection=None, parent=None):
    obj_copy = obj.copy()
    if data and obj_copy.data:
        obj_copy.data = obj_copy.data.copy()
    if actions and obj_copy.animation_data and obj_copy.animation_data.action:
        obj_copy.animation_data.action = obj_copy.animation_data.action.copy()
    if collection:
        collection.objects.link(obj_copy)
    if children:
        for child in obj.children:
            child_copy = duplicate_object(child, data, actions, children, collection, obj_copy)    
            child_copy.parent = obj_copy
            child_copy.matrix_parent_inverse = child.matrix_parent_inverse
        
    return obj_copy


##########################################################################################
def delete(obj_name):
    # Get the object by its name and delete if exists
    obj = bpy.data.objects.get(obj_name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)