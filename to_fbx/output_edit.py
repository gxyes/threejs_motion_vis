import bpy
import math
from mathutils import Matrix, Vector, Quaternion, Euler
from math import radians
import numpy as np
import os


# define
os.chdir("D:/threejs_motion_vis/to_fbx/")
smpl_bones = {
    0: "Pelvis",
    1: "L_Hip",
    2: "R_Hip",
    3: "Spine1",
    4: "L_Knee",
    5: "R_Knee",
    6: "Spine2",
    7: "L_Ankle",
    8: "R_Ankle",
    9: "Spine3",
    10: "L_Foot",
    11: "R_Foot",
    12: "Neck",
    13: "L_Collar",
    14: "R_Collar",
    15: "Head",
    16: "L_Shoulder",
    17: "R_Shoulder",
    18: "L_Elbow",
    19: "R_Elbow",
    20: "L_Wrist",
    21: "R_Wrist",
#    22: "L_Hand",
#    23: "R_Hand",
}

# functions
def rod_rigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat

def bone_execute():
    # change bone's name
    armature = bpy.data.armatures['Armature']
    
    # mapping
    bones_mixamo = {'Hips': 'Pelvis','LeftUpLeg': 'L_Hip','RightUpLeg': 'R_Hip','Spine2': 'Spine3','Spine1': 'Spine2','Spine': 'Spine1','LeftLeg': 'L_Knee','RightLeg': 'R_Knee','LeftFoot': 'L_Ankle','RightFoot': 'R_Ankle','LeftToeBase': 'L_Foot','RightToeBase': 'R_Foot','Neck': 'Neck','LeftShoulder': 'L_Collar','RightShoulder': 'R_Collar','Head': 'Head','LeftArm': 'L_Shoulder','RightArm': 'R_Shoulder','LeftForeArm': 'L_Elbow','RightForeArm': 'R_Elbow','LeftHand': 'L_Wrist','RightHand': 'R_Wrist'}
    for key in bones_mixamo:
        for bone in armature.bones:
            if key in bone.name:
                bone.name = bones_mixamo[key]
                break
            
    # Change bones' initial orientation
    object = bpy.context.edit_object
    
    bones = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot','Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
    bpy.ops.object.mode_set(mode='EDIT')
        
    for bone in bones:
        object.data.edit_bones[bone].use_connect = False
    for bone in bones:
        object.data.edit_bones[bone].tail[0] = object.data.edit_bones[bone].head[0]
        object.data.edit_bones[bone].tail[1] = object.data.edit_bones[bone].head[1] + 2
        object.data.edit_bones[bone].tail[2] = object.data.edit_bones[bone].head[2]
        object.data.edit_bones[bone].roll = 0

def process_pose(current_frame, pose, trans, pelvis_position):
    if pose.shape[0] == 72:
        rod_rots = pose.reshape(24, 3)
    else:
        rod_rots = pose.reshape(26, 3)

    mat_rots = [rod_rigues(rod_rot) for rod_rot in rod_rots]

    armature = bpy.data.objects['Armature']
    
    bones = armature.pose.bones
    bones[smpl_bones[0]].location = (
        Vector((100 * trans[1], 100 * trans[2], 100 * trans[0])) - pelvis_position
    )
    bones[smpl_bones[0]].keyframe_insert("location", frame=current_frame)

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 22:
            continue

        bone = bones[smpl_bones[index]]
        bone_rotation = Matrix(mat_rot).to_quaternion()
        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))

        if index == 0:
            # Rotate pelvis so that avatar stands upright and looks along negative Y avis
            bone.rotation_quaternion = (quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
        else:
            bone.rotation_quaternion = bone_rotation

        bone.keyframe_insert("rotation_quaternion", frame=current_frame)
    return

# npy to fbx
def process_poses(poses, gender):
    
    # basic info settings
    if gender == "female":
        model_path = female_model_path
        
    elif gender == "male":
        model_path = male_model_path
        
    # print basic info
    print(f"Gender: {gender}")
    print(f"Number of source poses: {str(poses.shape[0])}")
    print(f"Source frames-per-second: {str(fps_source)}")
    print(f"Target frames-per-second: {str(fps_target)}")
    print("--------------------------------------------------")

    # set up scene using blender
    scene = bpy.data.scenes["Scene"]
    scene.render.fps = fps_target

    # delete the default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    # .fbx template file
    bpy.ops.import_scene.fbx(filepath=model_path)

    sample_rate = int(fps_source / fps_target)
    scene.frame_end = (int)(poses.shape[0] / sample_rate)
    bpy.ops.object.mode_set(mode="EDIT")
    
    # change bone settings
    bone_execute()
    
    print("Done with changing bones")
    print("--------------------------------------------------")
    
    # process poses
    pelvis_position = Vector(
        bpy.context.edit_object.data.edit_bones[0].head
    )
    bpy.ops.object.mode_set(mode="OBJECT")
#    
#    pelvis_position = Vector(
#        bpy.context.edit_object.data.edit_bones["Pelvis"].head
#    )   
    
    source_index = 0
    frame = 1
    offset = np.array([0.0, 0.0, 0.0])
    trans = np.zeros((poses.shape[0], 3))
    
#    for i in range(poses.shape[0]):
#        trans[i] = [0, 0, i*0.01]
    
    print(trans)
    
    while source_index < poses.shape[0]:
        print("Adding poses: " + str(source_index))
        scene.frame_set(frame)
        process_pose(
            frame, poses[source_index], (trans[source_index] - offset), pelvis_position
        )
        source_index += sample_rate
        frame += 1

    return frame
 
def export_animated_mesh(output_path):
    print("here")
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if bpy.ops.object.select_all.poll():
        bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Armature"].select_set(True)
    bpy.data.objects["Armature"].children[0].select_set(True)

    ov=bpy.context.copy()
    ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]

#    bpy.ops.transform.resize(value=(1000.0, 1000.0, 1000.0), orient_type='LOCAL')
    
    # bpy.data.objects['Armature'].scale = (0.008, 0.008, 0.008)
    bpy.data.objects['Armature'].location = (0, 0, 1.04)
#    bpy.data.objects['Armature'].rotation = (90, 90, 180)
    
    bpy.ops.transform.rotate(
        value=math.pi/2, 
        orient_axis='Y', 
        orient_type='GLOBAL', 
        orient_matrix_type='GLOBAL', 
        constraint_axis=(False, True, False))
        
    bpy.ops.transform.rotate(
        value=math.pi/2, 
        orient_axis='Z', 
        orient_type='GLOBAL', 
        orient_matrix_type='GLOBAL', 
        constraint_axis=(False, False, True))

    if output_path.endswith(".glb"):
        print("Exporting to glTF binary (.glb)")
        # Currently exporting without shape/pose shapes for smaller file sizes
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format="GLB",
#            export_selected=True,
            export_morph=False,
        )
    elif output_path.endswith(".fbx"):
        print("Exporting to FBX binary (.fbx)")
        bpy.ops.export_scene.fbx(
            filepath=output_path, use_selection=True, add_leaf_bones=False
        )
    else:
        print("ERROR: Unsupported export format: " + output_path)
        sys.exit(1)

    return   

if __name__ == "__main__":
    # fps settings
    fps_source = 30
    fps_target = 30
    
    # female and male model settings
    female_model_path = "./fbx_templates/Vanguard.fbx"
    male_model_path = "./fbx_templates/Remy.fbx"

    # pose settings
    poses_path = "./acton.npy"
    poses = np.load(poses_path)

    # output path
    output_path = "./glb_models/Vanguard.glb"

    frame = process_poses(poses, gender="female")
    export_animated_mesh(output_path)