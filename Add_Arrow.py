import bpy
import sys
import math
import numpy as np
from mathutils import Vector

# Load local modules
dir_src = r"C:\Users\nbt571\Documents\Blender\Scripts"
dir_dep = r"C:\Users\nbt571\Documents\Blender\Scripts\dependencies"
for dir in [dir_src, dir_dep]:
    if not dir in sys.path:
        sys.path.append(dir)

import spherical_functions
import importlib
importlib.reload(spherical_functions)
from spherical_functions import sph2cart




# Function to create an arrow-like object
def create_arrow(longitude, latitude, arrow_length, cone_length=0.5, 
                 length_include_head=True, cylinder_radius=0.1, cone_radius=0.3):
    
    if length_include_head:
        arrow_length = arrow_length - cone_length
        
        
    # Direction vector in cartesian coordinates
    direction = sph2cart(longitude, latitude, arrow_length)
    direction = Vector(direction)
    
    
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=cylinder_radius, depth=arrow_length, location=(0, 0, arrow_length/2))
    cylinder_obj = bpy.context.active_object

    # Set pivot to global origin
    bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    # Rotate cylinder
    cylinder_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

    
    # Create cone
    bpy.ops.mesh.primitive_cone_add(radius1=cone_radius, depth=cone_length, location=(0, 0, arrow_length + cone_length/2))
    cone_obj = bpy.context.active_object
    
    # Set pivot to global origin
    bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    # Rotate cone
    cone_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    
    
    # Parent cone to cylinder
    cone_obj.parent = cylinder_obj
    cone_obj.matrix_parent_inverse = cylinder_obj.matrix_world.inverted()



# Parameters
arrow_length = 6.37    # Length of the arrow
longitude = 90   # Longitude angle in degrees
latitude = 0   # Latitude angle in degrees

# Create arrow
create_arrow(longitude, latitude, arrow_length)