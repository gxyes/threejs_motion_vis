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
    bpy.data.objects['Camera'].location = (3.9932, -4.235, 7.1698)
    bpy.data.objects['Camera'].rotation_euler = (radians(45), radians(0), radians(45))
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
        if i <= 9:
            avatar_name = "Armature.00" + str(i)
        else: 
            avatar_name = "Armature.0" + str(i)
#        avatar_name = "Armature.00" + str(i)
        if i == 0:
            avatar_name = "Armature"
        
        # i = i % 5
        # get armature
        armature = bpy.data.objects[avatar_name]
        # armature.location.y = i*0.6
        armature.location.z = 0.057978
        
        # get mesh
        meshs = []
        for child in armature.children:
            meshs.append(child)
        
        
        for mesh in meshs:
            mesh.active_material.blend_method = "HASHED"
            mesh.active_material.shadow_method = "HASHED"
            inputs = mesh.active_material.node_tree.nodes["Principled BSDF"].inputs  
#                inputs['Alpha'].default_value = 0.1 * (num_frames_show - abs(i-num_frames_show // 2))
            inputs[0].default_value[0] = (num_frames_show-i)/(num_frames_show+1)*(1-0.316) + 0.310
#            
        # set action offset
        offset = sample_rate*i
        print(offset)
        
        # get action
        action = armature.animation_data.action
        for fc in action.fcurves:
            for kf in fc.keyframe_points:
                kf.co.x += offset
            
    bpy.context.scene.frame_set(119)
 
def change_model_x(pos_x): 
    for i in range(num_frames_show):
        avatar_name = "Armature.00" + str(i+5)
        
        armature = bpy.data.objects[avatar_name]
        armature.location.x = pos_x
                          
    
if __name__ == "__main__":
    num_frames_show = 8
    total_frames = 120
    pos_x = 2
    
    # background color
#    inputs = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    
    bpy.context.scene.world.color = (0.0871424, 0.0871424, 0.0871424)
    
    model_path_remodiffuse = "./glb_models/Michelle.glb"
    model_path_mdm = "./glb_models/XBot_A_person_skips_in_a_circle.glb"
    
    ground_img_path = "D:/threejs_motion_vis/to_fbx/img/checkerboard_more.png" 
    
    # scene_initialize
    scene_init(image_path=ground_img_path)
    
    # add avatars for remodiffuse
#    for i in range(num_frames_show):
#        add_avatar(model_path=model_path_remodiffuse, avatar_index=i)
#        
    # add avatars for mdm
    for i in range(num_frames_show):
        add_avatar(model_path=model_path_mdm, avatar_index=i)

    # edit position & animations
    edit_avatars(num_frames_show, total_frames)
    
    # move one set of models by pos_x
    # change_model_x(pos_x)
