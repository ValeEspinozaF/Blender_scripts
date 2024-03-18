# Load public modules 
import bpy
import os
import sys
import pandas as pd
import numpy as np


# Load local modules
dir_src = r"C:\Users\nbt571\Documents\Blender\Scripts"
dir_dep = r"C:\Users\nbt571\Documents\Blender\Scripts\dependencies"
for dir in [dir_src, dir_dep]:
    if not dir in sys.path:
        sys.path.append(dir)
    
import Add_hollowSphere
import importlib
importlib.reload(Add_hollowSphere)
from Add_hollowSphere import hollow_sphere

import Add_wrappedContour
import importlib
importlib.reload(Add_wrappedContour)
from Add_wrappedContour import wrappedContour


radius = 6.370


### ADD HOLLOW SPHERE ####

u, v = 16, 10
radius_outer, radius_inner = radius, radius*0.9
hollow_sphere(radius_outer, radius_inner, u, v, color_tuple=(0.5, 0.5, 0.5, 0.3))
bpy.ops.object.shade_smooth()



### ADD PLATE ###

# Load contour coordinates
inputs_dir = r"C:\Users\nbt571\Documents\Blender\Scripts\test_inputs"
cntr_sa_path = os.path.join(inputs_dir, "BDR_SA.txt")
cntr_68_path = os.path.join(inputs_dir, "CNTR68_STGs_0_2_SA.txt")
cntr_df = pd.read_csv(cntr_sa_path, delimiter=' ', header=None, names=['lon', 'lat'])

# Wrap contour to sphere
verts, edges, faces = wrappedContour(cntr_df, bpy.data.objects['sphere_outer'], radius*1.03)

# Add Contour to scene
mesh_data = bpy.data.meshes.new("plate_data")
mesh_data.from_pydata(verts, edges, faces)
mesh_obj = bpy.data.objects.new("plate_object", mesh_data)
bpy.context.collection.objects.link(mesh_obj)

# Add color
matb = bpy.data.materials.new("Blue")
matb.diffuse_color = (0,0,1,1)
bpy.data.objects['plate_object'].active_material = matb