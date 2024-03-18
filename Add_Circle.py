import bpy
import sys
import math
import numpy as np
from mathutils import Quaternion

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


# Function to convert Cartesian coordinates to Euler angles
def cartesian_to_euler(normal_vector):
    
    normal_vector = np.radians(normal_vector)
    
    # Calculate rotations around the main axes XYZ
    x = math.atan2(normal_vector[1], normal_vector[2])
    y = math.atan2(-normal_vector[0], math.sqrt(normal_vector[1]**2 + normal_vector[2]**2))
    z = 0  # No rotation around Z-axis
    
    return (x, y, z)


def rotation_matrix_transpose(vector):
    
    # Normalize the vector
    vector = np.radians(vector)
    norm_vector = vector / np.linalg.norm(vector)
    
    # Calculate the cross product with the target direction (0, 0, 0)
    cross_product = np.cross(norm_vector, [0, 0, 0])
    
    # Calculate the dot product with the target direction (0, 0, 0)
    dot_product = np.dot(norm_vector, [0, 0, 0])
    
    # Construct the rotation matrix transpose
    rotation_matrix_transpose = np.array([
        [dot_product, -cross_product[2], cross_product[1]],
        [cross_product[2], dot_product, -cross_product[0]],
        [-cross_product[1], cross_product[0], dot_product]
    ])
    
    return rotation_matrix_transpose


# Function to create a circular surface
def create_circle_surface(radius, longitude, latitude, segments=32, color=(0.0, 0.0, 1.0, 1.0)):
    
    # Add circle mesh
    bpy.ops.mesh.primitive_circle_add(radius=radius, fill_type='NGON', location=(0, 0, 0), vertices=segments)    
    circle_object = bpy.context.active_object
    normal_vector = circle_object.rotation_euler.to_matrix() @ bpy.data.objects['Circle'].matrix_world.to_3x3().col[2]
    
    # Create a new material
    material = bpy.data.materials.new(name="CircleMaterial")
    material.diffuse_color = color
    
    if circle_object.data.materials:
        # Assign to first material slot
        circle_object.data.materials[0] = material
    else:
        # No material slots
        circle_object.data.materials.append(material)
    
    
    # Set the orientation of the circle surface
    circle_object.rotation_mode = 'XZY'
    circle_object.rotation_euler.rotate_axis('Y', np.radians(90))
    circle_object.rotation_euler.rotate_axis('Y', -np.radians(latitude))
    circle_object.rotation_euler.rotate_axis('X', -np.radians(longitude))
    
    return circle_object


# Parameters
radius = 6.370  # Radius of the circle surface
longitude = 20  # Longitude angle in degrees
latitude = -2   # Latitude angle in degrees
segments = 16
color = (1.0, 0.0, 0.0, 1.0)

# Create circle surface
circle_surface = create_circle_surface(radius, longitude, latitude, segments, color)