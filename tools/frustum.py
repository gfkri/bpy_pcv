import bpy
import bmesh
import mathutils
import numpy as np


#######################################################################################################################
def get_camera_frustum_corners(camera, farplane_ratio=1.0, nearplane_ratio=0.0):
    """ Get the corners of the camera's frustum in world space.
    
         2+-----+6
         /|    /|
       3+-----+7|
        | |   | |
        |0+---|-+4
        |/    |/
       1+-----+5
    
    Parameters
    ----------
    camera : bpy.types.Object
        The camera object for which the frustum corners will be returned.
    farplane_ratio : float, optional
        The ratio of the far plane distance to the camera's far clipping distance (default is 1.0).
    nearplane_ratio : float, optional
        The ratio of the near plane distance to the camera's near clipping distance (default is 0.0).
    
    Returns
    -------
    list of mathutils.Vector
        The corners of the camera's frustum in world space.
    """
    
    assert farplane_ratio >= 0.0 and nearplane_ratio >= 0.0, "The farplane_ratio and nearplane_ratio must be greater or equal to 0.0"
    
    # Ensure a valid camera is selected
    cam_data = camera.data
    scene = bpy.context.scene

    # Get the camera's world position
    camera_origin = camera.matrix_world.translation

    # Get the near plane corners in world space
    near_plane_corners = [camera.matrix_world @ corner for corner in cam_data.view_frame(scene=scene)]
    
    frustum_corners = []
    for corner in near_plane_corners:
        frustum_corners.append(camera_origin + (corner - camera_origin) * nearplane_ratio)
        frustum_corners.append(camera_origin + (corner - camera_origin) * farplane_ratio)
        
    # Reorder the corners to match the order of a blender cube
    frustum_corners = [frustum_corners[idx] for idx in [5, 4, 7, 6, 3, 2, 1, 0]]
        
    return [np.array(c) for c in frustum_corners]


#######################################################################################################################
def get_cube_vertices(center, x_axis, y_axis, z_axis, size):
    """ Place the vertices of a cuboid with the given center, axes and size.
    
    Parameters
    ----------
    center : mathutils.Vector
        The center of the cuboid.
    x_axis : mathutils.Vector
        The x-axis of the cuboid.
    y_axis : mathutils.Vector
        The y-axis of the cuboid.
    z_axis : mathutils.Vector
        The z-axis of the cuboid.
    size : mathutils.Vector
        The size of the cuboid.
    
    Returns
    -------
    list of mathutils.Vector
        The vertices of the cuboid.
    """
    
    hs = size / 2
    if type(hs) == float or type(hs) == int:
        hs = np.array((hs, hs, hs))    
    
    vertices = [
        center - hs[0] * x_axis - hs[1] * y_axis - hs[2] * z_axis,
        center - hs[0] * x_axis - hs[1] * y_axis + hs[2] * z_axis,
        center - hs[0] * x_axis + hs[1] * y_axis - hs[2] * z_axis,
        center - hs[0] * x_axis + hs[1] * y_axis + hs[2] * z_axis,
        center + hs[0] * x_axis - hs[1] * y_axis - hs[2] * z_axis,
        center + hs[0] * x_axis - hs[1] * y_axis + hs[2] * z_axis,
        center + hs[0] * x_axis + hs[1] * y_axis - hs[2] * z_axis,
        center + hs[0] * x_axis + hs[1] * y_axis + hs[2] * z_axis,
    ]

    return vertices


#######################################################################################################################
def place_wireframe_cube(bm, center, x_axis, y_axis, z_axis, size):
    cube_vertices = get_cube_vertices(center, x_axis, y_axis, z_axis, size)
    cube_vertices = [bm.verts.new(c) for c in cube_vertices]
    
    for i in range(4):
        bm.edges.new((cube_vertices[i], cube_vertices[i + 4]))
        
    left_face_vertices = get_left(cube_vertices)  
    for i in range(4):
        bm.edges.new((left_face_vertices[i], left_face_vertices[(i + 1) % 4]))
      
    right_face_vertices = get_right(cube_vertices)
    for i in range(4):
        bm.edges.new((right_face_vertices[i], right_face_vertices[(i + 1) % 4]))
        
    return cube_vertices


#######################################################################################################################
def get_front(v):
    return [v[i] for i in [1, 5, 7, 3]]

def get_back(v):
    return [v[i] for i in [2, 6, 4, 0]]

def get_top(v):
    return [v[i] for i in [2, 3, 7, 6]]

def get_bottom(v):
    return [v[i] for i in [4, 5 ,1, 0]]

def get_right(v):
    return [v[i] for i in [4, 6, 7, 5]]

def get_left(v):
    return [v[i] for i in [1, 3, 2, 0]]


#######################################################################################################################
def create_frustum(name, frustum_corners, thickness=0.025):
    """ Draw a frustum pyramid in Blender. 
    
    Parameters
    ----------
    name : str
        The name of the frustum object.
    frustum_corners : list of mathutils.Vector
        The corners of the frustum pyramid.
    thickness : float, optional
        The thickness of the frustum edges (default is 0.025).

    """        
    x_axis = np.array(frustum_corners[7] - frustum_corners[3])
    y_axis = np.array(frustum_corners[7] - frustum_corners[5])
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    
    # Create a new mesh for the frustum pyramid
    mesh = bpy.data.meshes.new("FrustumPyramid")
    obj = bpy.data.objects.new(name, mesh)
    # bpy.context.collection.objects.link(obj)

    # Create the pyramid using bmesh
    bm = bmesh.new()
    
    corner_vertex_list = []
    
    for fr_crn in frustum_corners:
        corner_cube_vertices = place_wireframe_cube(bm, fr_crn, x_axis, y_axis, z_axis, thickness)
        corner_vertex_list.append(corner_cube_vertices)
    
    bm.verts.ensure_lookup_table()
                
    # add missing faces from corner cubes
    for i in range(4):         
        bm.faces.new(get_back(corner_vertex_list[i * 2]))
        bm.faces.new(get_front(corner_vertex_list[i * 2 + 1]))
        
        bm.faces.new(get_left(corner_vertex_list[i]))
        bm.faces.new(get_right(corner_vertex_list[i + 4]))
        
        bm.faces.new(get_bottom(corner_vertex_list[i + (i // 2) * 2]))    
        bm.faces.new(get_top(corner_vertex_list[i + (i // 2) * 2 + 2]))
    
    
    def connect_cubes(vertices1, vertices2):
        l = len(vertices1)
        for idx in range(l):
            # bm.edges.new((vertices1[idx], vertices2[l - idx - 1]))
            bm.faces.new([vertices1[idx], vertices1[(idx + 1) % l], 
                          vertices2[(l - idx - 2) % l], vertices2[l - idx - 1]])
    
    # near plane corner connections    
    connect_cubes(get_bottom(corner_vertex_list[7]), get_top(corner_vertex_list[5]))
    connect_cubes(get_left(corner_vertex_list[5]), get_right(corner_vertex_list[1]))  
    connect_cubes(get_top(corner_vertex_list[1]), get_bottom(corner_vertex_list[3]))  
    connect_cubes(get_right(corner_vertex_list[3]), get_left(corner_vertex_list[7]))  
    
    # # far plane corner cornnections
    connect_cubes(get_bottom(corner_vertex_list[6]), get_top(corner_vertex_list[4]))
    connect_cubes(get_left(corner_vertex_list[4]), get_right(corner_vertex_list[0]))  
    connect_cubes(get_top(corner_vertex_list[0]), get_bottom(corner_vertex_list[2]))  
    connect_cubes(get_right(corner_vertex_list[2]), get_left(corner_vertex_list[6]))  
    
    for i in range(4):
        connect_cubes(get_front(corner_vertex_list[i * 2]), get_back(corner_vertex_list[i * 2 + 1]))
    
    # Write the geometry to the mesh
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    
    print("Frustum pyramid created successfully.")
    return obj
     

#######################################################################################################################
if __name__ == "__main__":
    camera = bpy.context.scene.camera
    if not camera or camera.type != 'CAMERA':
        print("Select a valid camera in the scene")
    else:
        create_frustum(camera)


