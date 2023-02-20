import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
import bpy
from math import radians
import os
import sys
# from bones import ybot_bones, smpl_bones


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
    22: "L_Hand",
    23: "R_Hand",
}
# f_prefix = "f_avg_"
# f_prefix = "m_avg_"
f_prefix = ""
# f_prefix = "mixamorig:"


def rod_rigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


def setup_scene(model_path, fps_target):
    scene = bpy.data.scenes["Scene"]
    scene.render.fps = fps_target

    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    # .fbx template file
    bpy.ops.import_scene.fbx(filepath=model_path)


def process_poses(poses, gender):
    trans = np.zeros((poses.shape[0], 3))
    if gender == "female":
        model_path = female_model_path
        for k, v in bone_name_from_index.items():
            bone_name_from_index[k] = f_prefix + v
            # print(bone_name_from_index[k])
    elif gender == "male":
        model_path = male_model_path
        for k, v in bone_name_from_index.items():
            bone_name_from_index[k] = f_prefix + v
            # print(bone_name_from_index[k])

    print(f"Gender: {gender}")
    print(f"Number of source poses: {str(poses.shape[0])}")
    print(f"Source frames-per-second: {str(fps_source)}")
    print(f"Target frames-per-second: {str(fps_target)}")
    print("--------------------------------------------------")

    setup_scene(model_path, fps_target)

    scene = bpy.data.scenes["Scene"]
    sample_rate = int(fps_source / fps_target)
    scene.frame_end = (int)(poses.shape[0] / sample_rate)
    bpy.ops.object.mode_set(mode="EDIT")

    print("============================")
    print(bpy.data.armatures[0].edit_bones.keys())

    pelvis_position = Vector(
        bpy.data.armatures[0].edit_bones[bone_name_from_index[0]].head
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    source_index = 0
    frame = 1

    offset = np.array([0.0, 0.0, 0.0])

    while source_index < poses.shape[0]:
        print("Adding poses: " + str(source_index))
        scene.frame_set(frame)
        process_pose(
            frame, poses[source_index], (trans[source_index] - offset), pelvis_position
        )
        source_index += sample_rate
        frame += 1

    return frame


def process_pose(current_frame, pose, trans, pelvis_position):
    if pose.shape[0] == 72:
        rod_rots = pose.reshape(24, 3)
    else:
        rod_rots = pose.reshape(26, 3)

    mat_rots = [rod_rigues(rod_rot) for rod_rot in rod_rots]

    armature = bpy.data.objects["Armature"]
    bones = armature.pose.bones
    bones[bone_name_from_index[0]].location = (
        Vector((100 * trans[1], 100 * trans[2], 100 * trans[0])) - pelvis_position
    )
    bones[bone_name_from_index[0]].keyframe_insert("location", frame=current_frame)

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue

        bone = bones[bone_name_from_index[index]]
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


def export_animated_mesh(output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Armature"].select_set(True)
    bpy.data.objects["Armature"].children[0].select_set(True)

    ov=bpy.context.copy()
    ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
    # bpy.ops.transform.rotate(ov, value=3.14)
    bpy.ops.transform.resize(value=(1000.0, 1000.0, 1000.0), orient_type='LOCAL')
    bpy.data.objects['Armature'].location = (0, 0, 3)
    # bpy.ops.transform.rotate(
    #     value=0.349066, 
    #     orient_axis='Y', 
    #     orient_type='GLOBAL', 
    #     orient_matrix_type='GLOBAL', 
    #     constraint_axis=(False, True, False))

    if output_path.endswith(".glb"):
        print("Exporting to glTF binary (.glb)")
        # Currently exporting without shape/pose shapes for smaller file sizes
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format="GLB",
            export_selected=True,
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
    fps_source = 30
    fps_target = 30
    female_model_path = "./SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"
    # female_model_path = "./amy.fbx"

    # male_model_path = "./SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"
    male_model_path = "./obama.fbx"
    # bone_name_from_index = ybot_bones
    bone_name_from_index = smpl_bones

    poses_path = "./acton.npy"
    poses = np.load(poses_path)

    output_path = "./obamatry.glb"

    frame = process_poses(poses, gender="male")
    export_animated_mesh(output_path)
