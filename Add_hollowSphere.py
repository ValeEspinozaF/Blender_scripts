
import bpy
import bmesh


def hollow_sphere(radius_outer, radius_inner, u, v, color_tuple=(0.5, 0.5, 0.5, 0.3)):
    
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=radius_outer)
    mesh = bpy.data.meshes.new("sphere_data_outer")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("sphere_outer", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.shade_smooth()

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=radius_inner)
    mesh = bpy.data.meshes.new("sphere_data_inner")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("sphere_inner", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.shade_smooth()


    # Select the inner sphere and set it as the active object
    bpy.context.view_layer.objects.active = bpy.data.objects['sphere_inner']
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, 0.2)})
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create a new material
    mat = bpy.data.materials.new(name="NewMaterial")
    mat.use_nodes = False
    mat.diffuse_color = color_tuple
    
    # Assign the material to the outer sphere
    bpy.context.view_layer.objects.active = bpy.data.objects['sphere_outer']
    bpy.context.active_object.data.materials.append(mat)
    
    # Delete the inner sphere
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects['sphere_inner']
    bpy.context.active_object.select_set(True)
    bpy.ops.object.delete()
