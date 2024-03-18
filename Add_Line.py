import bpy
import sys
import math

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


# Function to create an arrow-like thin tube
def create_arrow(length, longitude, latitude, objectName='ArrowObject'):

    end_point = sph2cart(longitude, latitude, length)
    
    # Create curve object
    curve_data = bpy.data.curves.new(name='ArrowCurve', type='CURVE')
    curve_data.dimensions = '3D'
    
    # Create spline
    spline = curve_data.splines.new(type='NURBS')
    spline.points.add(1)
    spline.points[0].co = (0, 0, 0, 1)
    spline.points[1].co = end_point + (1,)
    
    # Create object
    curve_object = bpy.data.objects.new(objectName, curve_data)
    bpy.context.collection.objects.link(curve_object)
    
    # Add material to the curve object and set its color
    material = bpy.data.materials.new(name="RedMaterial")
    material.diffuse_color = (1, 0, 0, 1)  # Red color
    curve_object.data.materials.append(material)
    
    

# Parameters
length = 7.0  # Length of the arrow
longitude = -45  # Longitude angle in degrees
latitude = -16   # Latitude angle in degrees

# Create arrow
create_arrow(length, longitude, latitude)