import bpy
import math
import addon_utils
from mathutils import Matrix, Vector, Quaternion, Euler
from math import radians
import numpy as np
import os


os.chdir("D:/threejs_motion_vis/to_fbx/")

def scene_init(image_path): 
    # set up scene using blender
    scene = bpy.data.scenes["Scene"]
        
    # Light & Camera
    bpy.data.objects['Camera'].location = (11, -6, 2.76)
    bpy.data.objects['Camera'].rotation_euler = (radians(80), radians(0), radians(60))
    bpy.data.objects['Light'].location = (1, -3, 6)

    # delete the default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()
        
    # add a plane
    addon_name = "io_import_images_as_planes"
    loaded_default, load_state = addon_utils.check(addon_name)
    if not load_state:
        print("here")
        addon_utils.enable(addon_name)
   
    bpy.ops.import_image.to_plane(files=[{"name": image_path}])

    bpy.data.objects['checkerboard_more'].rotation_euler = (radians(180), radians(0), radians(0))
    bpy.data.objects['checkerboard_more'].scale = (50, 50, 50)
        

def add_avatar(model_path, avatar_index):
    # .fbx template file
    bpy.ops.import_scene.gltf(filepath=model_path)
    
def edit_avatars(num_frames_show, total_frames):
    sample_rate = total_frames // num_frames_show
    for i in range(num_frames_show):
        avatar_name = "Armature.00" + str(i)
        mesh_name = "Ch03.00" + str(i)
        sample_frame = i*sample_rate + 1
        
        if i == 0:
            avatar_name = "Armature"
            mesh_name = "Ch03"
        
        # get armature
        armature = bpy.data.objects[avatar_name]
        armature.location.y = i*0.6
       
        offset = sample_rate*i
        
        # get action
        action = bpy.data.actions[i]
        for fc in action.fcurves:
            for kf in fc.keyframe_points:
                kf.co.x += offset

        # get mesh
        mesh = bpy.data.objects[mesh_name]
        print("here")
        
        if i != num_frames_show // 2:
            mesh.active_material.blend_method = "HASHED"
            mesh.active_material.shadow_method = "HASHED"
            
            inputs = mesh.active_material.node_tree.nodes["Principled BSDF"].inputs
            
            inputs['Alpha'].default_value = 0.1 * (num_frames_show - abs(i-num_frames_show // 2))
            
    bpy.context.scene.frame_set(60)
   
    
    
if __name__ == "__main__":
    num_frames_show = 5
    total_frames = 60
    model_path = "./glb_models/Michelle.glb"
    ground_img_path = "D:/threejs_motion_vis/pages/imgs/checkerboard_more.png" 
    
    # scene_initialize
    scene_init(image_path=ground_img_path)
    
    # add avatars
    for i in range(num_frames_show):
        add_avatar(model_path=model_path, avatar_index=i)

    # edit position & animations
    edit_avatars(num_frames_show, total_frames)