import numpy as np
import os


class MotionData():
    def __init__(self, root_pos, root_rot, dof_pos, root_lin_vel, root_ang_vel, dof_vel, key_pos, motion_fps):
        self.root_pos = root_pos
        self.root_rot = root_rot
        self.dof_pos = dof_pos
        self.root_lin_vel = root_lin_vel  # local
        self.root_ang_vel = root_ang_vel  # local
        self.dof_vel = dof_vel
        self.key_pos = key_pos  # local
        self.motion_fps = motion_fps

    @classmethod
    def load_from_file(cls, file_path):

        motion_data = np.load(file_path, allow_pickle=True)
        root_pos = motion_data['root_pos']
        root_rot = motion_data['root_rot']
        dof_pos = motion_data['dof_pos']
        root_lin_vel = motion_data['root_lin_vel']  # local
        root_ang_vel = motion_data['root_ang_vel']  # local
        dof_vel = motion_data['dof_vel']
        key_pos = motion_data['key_pos']  # local
        motion_fps = motion_data['motion_fps']

        return cls(root_pos, root_rot, dof_pos, root_lin_vel, root_ang_vel, dof_vel, key_pos, motion_fps)


MOTION_NUM = "07_08"
MOTION_FILE_PATH = os.path.join(
    "./result/output/isaac_motion", MOTION_NUM + ".npz")

motion_data = MotionData.load_from_file(MOTION_FILE_PATH)

print(motion_data.root_pos.shape)
print(motion_data.motion_fps)
print(motion_data.key_pos.shape)
