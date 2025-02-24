import bpy
from mathutils import Vector
import math

utils = bpy.data.texts["utils.py"].as_module()

##########################################################################################
def create_bounding_box(pos, dim, heading, bb_obj_name='Bounding Box', thickness=0.05, collection_name='Bounding Boxes'):
    '''Creates a bounding box around an object defined by position, dimension and heading;
       assumes the objects origin in the center of the object's ground plane'''
    t = thickness
    h = heading * 180.0 / math.pi
    l2 = dim[0] / 2.0
    w2 = dim[1] / 2.0
    h2 = dim[2] / 2.0  
    
    properties = [
        {'scale': (l2+t, t, t), 'translation': (0, w2, h2)},
        {'scale': (l2+t, t, t), 'translation': (0, -w2, h2)},
        {'scale': (l2+t, t, t), 'translation': (0, w2, -h2)},
        {'scale': (l2+t, t, t), 'translation': (0, -w2, -h2)},
        {'scale': (t, t, h2-t), 'translation': (l2, -w2, 0)},
        {'scale': (t, t, h2-t), 'translation': (-l2, -w2, 0)},
        {'scale': (t, t, h2-t), 'translation': (l2, w2, 0)},
        {'scale': (t, t, h2-t), 'translation': (-l2, w2, 0)},
        {'scale': (t, w2-t, t), 'translation': (l2, 0, -h2)},
        {'scale': (t, w2-t, t), 'translation': (-l2, 0, -h2)},
        {'scale': (t, w2-t, t), 'translation': (l2, 0, h2)},
        {'scale': (t, w2-t, t), 'translation': (-l2, 0, h2)},
    ]

    # Calculate the bounding box coordinates
#    bounding_box_coords = [obj.matrix_world @ Vector(coord) for coord in obj.bound_box]
    
    # Move the pillars to a new collection for better organization
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[collection_name]
        
    created_pillars = []
    # Create the pillars (cubes) at the corners
    for prop in properties:
        bpy.ops.mesh.primitive_cube_add(size=1, location=Vector(prop['translation']) + Vector((0.0, 0.0, h2+t/2)))
        pillar = bpy.context.active_object
        pillar.scale = [2*el for el in prop['scale']]
               
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        pillar.rotation_euler.z = heading 
        pillar.location = pos
                
        if pillar.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(pillar)
        if pillar.name not in collection.objects:
            collection.objects.link(pillar)
        created_pillars.append(pillar)
        
    bpy.ops.object.select_all(action='DESELECT')
    # Select all objects in the "Pillars" collection
    for pillar in created_pillars:
        pillar.select_set(True)
        
    bpy.context.view_layer.objects.active = created_pillars[0]
    bpy.ops.object.join()    
    new_bb_obj = bpy.context.active_object

    orig_bb_obj = bpy.data.objects.get(bb_obj_name)
    if orig_bb_obj:
        print(f'Bounding Box {bb_obj_name} already exists, copying its attributes ...')
        if orig_bb_obj.animation_data:
            new_bb_obj.animation_data_create()
            new_bb_obj.animation_data.action = orig_bb_obj.animation_data.action.copy()
        
        if orig_bb_obj.modifiers:
            for orig_modifier in orig_bb_obj.modifiers:
                if orig_modifier.type == 'NODES':
                    modifier = new_bb_obj.modifiers.new(name=orig_modifier.name, type='NODES')
                    modifier.node_group = orig_modifier.node_group
        
        if orig_bb_obj.data.materials:             
            new_bb_obj.data.materials.clear()
            for material in orig_bb_obj.data.materials:
                new_bb_obj.data.materials.append(material)
    
      
    utils.delete(bb_obj_name)        
    new_bb_obj.name = bb_obj_name
    print(f'Created Bounding Box {bb_obj_name}')


##########################################################################################
def create_bounding_boxex():
    print('Creating bounding boxes ...')
    obj_names = ['beetle', 'touareg', 'car', 'van', 'bus']
    # obj_names = ['Cone']
    for obj_name in obj_names:  
        obj = bpy.data.objects[obj_name]
        pos = obj.location
        dim = obj.dimensions
        heading = obj.rotation_euler.z

        create_bounding_box(pos, dim, heading)

    print('Done')


##########################################################################################
if __name__ == '__main__':
#    create_bounding_boxex()
    create_bounding_box(Vector((0, 0, -0.5)), Vector((1, 1, 1)), 0, thickness=0.01)
