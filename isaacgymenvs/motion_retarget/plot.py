
import numpy as np
from matplotlib import pyplot as plt
import config.bvh_cfg.bvh_cmu_config as bvh_cfg
from scipy import stats

raw_motion_data = np.load('./result/temp/raw_motion_data.npy')
filtered_motion_data = np.load('./result/temp/filtered_motion_data.npy')

raw_base_pos = raw_motion_data[:, 0:3]
raw_base_quat = raw_motion_data[:, 3:7]
raw_dof_pos = raw_motion_data[:, 7:23]
raw_base_lin_vel = raw_motion_data[:, 23:26]
raw_base_ang_vel = raw_motion_data[:, 26:29]
raw_dof_vel = raw_motion_data[:, 29:45]
raw_end_effector_pos = raw_motion_data[:, 45:57]

filtered_base_pos = filtered_motion_data[:, 0:3]
filtered_base_quat = filtered_motion_data[:, 3:7]
filtered_dof_pos = filtered_motion_data[:, 7:23]
filtered_base_lin_vel = filtered_motion_data[:, 23:26]
filtered_base_ang_vel = filtered_motion_data[:, 26:29]
filtered_dof_vel = filtered_motion_data[:, 29:45]
filtered_end_effector_pos = filtered_motion_data[:, 45:57]
# filtered_end_effector_vel = filtered_motion_data[:, 57:69]


def main():

    t = np.arange(raw_motion_data.shape[0]) * bvh_cfg.FRAME_DURATION

    plot_base_pos(raw_base_pos, filtered_base_pos, t)
    plot_base_quat(raw_base_quat, filtered_base_quat, t)
    plot_base_lin_vel(raw_base_lin_vel, filtered_base_lin_vel, t)
    plot_base_ang_vel(raw_base_ang_vel, filtered_base_ang_vel, t)

    plot_dof_pos(raw_dof_pos, filtered_dof_pos, t)
    plot_dof_vel(raw_dof_vel, filtered_dof_vel, t)

    plot_end_effector_pos(raw_end_effector_pos, filtered_end_effector_pos, t)
    # plot_end_effector_vel(raw_end_effector_vel, filtered_end_effector_vel, t)

    plt.show()
    return


# -----------------------------------plot funciton----------------------------------- #

def plot_base_pos(raw_base_pos, filtered_base_pos, t):
    fig_base_pos = plt.figure()
    plt.subplot(311)
    plt.plot(t, raw_base_pos[:, 0], t, filtered_base_pos[:, 0])
    plt.title('base_pos')

    plt.subplot(312)
    # plt.plot(t, raw_base_pos[:, 1], t, filtered_base_pos[:, 1])
    plt.plot(t, raw_base_pos[:, 1], t, filtered_base_pos[:, 1])

    plt.subplot(313)
    plt.plot(t, raw_base_pos[:, 2], t, filtered_base_pos[:, 2])


def plot_base_quat(raw_base_quat, filtered_base_quat, t):
    fig_base_quat = plt.figure()
    plt.subplot(411)
    plt.plot(t, raw_base_quat[:, 0], t, filtered_base_quat[:, 0])
    plt.title('base_quat')

    plt.subplot(412)
    plt.plot(t, raw_base_quat[:, 1], t, filtered_base_quat[:, 1])

    plt.subplot(413)
    plt.plot(t, raw_base_quat[:, 2], t, filtered_base_quat[:, 2])

    plt.subplot(414)
    plt.plot(t, raw_base_quat[:, 3], t, filtered_base_quat[:, 3])


def plot_base_lin_vel(raw_base_lin_vel, filtered_base_lin_vel, t):
    fig_base_lin_vel = plt.figure()
    plt.subplot(311)
    plt.plot(t, raw_base_lin_vel[:, 0], t, filtered_base_lin_vel[:, 0])
    plt.title('base_lin_vel')

    plt.subplot(312)
    # plt.plot(t, raw_base_lin_vel[:, 1], t, filtered_base_lin_vel[:, 1])
    plt.plot(t, raw_base_lin_vel[:, 1], t, filtered_base_lin_vel[:, 1])

    plt.subplot(313)
    plt.plot(t, raw_base_lin_vel[:, 2], t, filtered_base_lin_vel[:, 2])


def plot_base_ang_vel(raw_base_ang_vel, filtered_base_ang_vel, t):
    fig_base_ang_vel = plt.figure()
    plt.subplot(311)
    plt.plot(t, raw_base_ang_vel[:, 0], t, filtered_base_ang_vel[:, 0])
    plt.title('base_ang_vel')

    plt.subplot(312)
    # plt.plot(t, raw_base_ang_vel[:, 1], t, filtered_base_ang_vel[:, 1])
    plt.plot(t, raw_base_ang_vel[:, 1], t, filtered_base_ang_vel[:, 1])

    plt.subplot(313)
    plt.plot(t, raw_base_ang_vel[:, 2], t, filtered_base_ang_vel[:, 2])


def plot_dof_pos(raw_dof_pos, filtered_dof_pos, t):
    fig_dof_pos_left = plt.figure()
    plt.subplot(811)
    plt.plot(t, raw_dof_pos[:, 0], t, filtered_dof_pos[:, 0])
    plt.title('dof_pos_left')

    plt.subplot(812)
    plt.plot(t, raw_dof_pos[:, 1], t, filtered_dof_pos[:, 1])

    plt.subplot(813)
    plt.plot(t, raw_dof_pos[:, 2], t, filtered_dof_pos[:, 2])

    plt.subplot(814)
    plt.plot(t, raw_dof_pos[:, 3], t, filtered_dof_pos[:, 3])

    plt.subplot(815)
    plt.plot(t, raw_dof_pos[:, 4], t, filtered_dof_pos[:, 4])

    plt.subplot(816)
    plt.plot(t, raw_dof_pos[:, 10], t, filtered_dof_pos[:, 10])

    plt.subplot(817)
    plt.plot(t, raw_dof_pos[:, 11], t, filtered_dof_pos[:, 11])
    # plt.plot(t, raw_dof_pos[:, 11], t, filtered_base_quat[:, 11])

    plt.subplot(818)
    plt.plot(t, raw_dof_pos[:, 12], t, filtered_dof_pos[:, 12])

    fig_dof_pos_right = plt.figure()
    plt.subplot(811)
    plt.plot(t, raw_dof_pos[:, 5], t, filtered_dof_pos[:, 5])
    plt.title('dof_pos_right')

    plt.subplot(812)
    plt.plot(t, raw_dof_pos[:, 6], t, filtered_dof_pos[:, 6])

    plt.subplot(813)
    plt.plot(t, raw_dof_pos[:, 7], t, filtered_dof_pos[:, 7])

    plt.subplot(814)
    plt.plot(t, raw_dof_pos[:, 8], t, filtered_dof_pos[:, 8])

    plt.subplot(815)
    plt.plot(t, raw_dof_pos[:, 9], t, filtered_dof_pos[:, 9])

    plt.subplot(816)
    plt.plot(t, raw_dof_pos[:, 13], t, filtered_dof_pos[:, 13])

    plt.subplot(817)
    plt.plot(t, raw_dof_pos[:, 14], t, filtered_dof_pos[:, 14])

    plt.subplot(818)
    plt.plot(t, raw_dof_pos[:, 15], t, filtered_dof_pos[:, 15])


def plot_dof_vel(raw_dof_vel, filtered_dof_vel, t):
    fig_dof_vel_left = plt.figure()
    plt.subplot(811)
    plt.plot(t, raw_dof_vel[:, 0], t, filtered_dof_vel[:, 0])
    plt.title('dof_vel_left')

    plt.subplot(812)
    plt.plot(t, raw_dof_vel[:, 1], t, filtered_dof_vel[:, 1])

    plt.subplot(813)
    plt.plot(t, raw_dof_vel[:, 2], t, filtered_dof_vel[:, 2])

    plt.subplot(814)
    plt.plot(t, raw_dof_vel[:, 3], t, filtered_dof_vel[:, 3])

    plt.subplot(815)
    plt.plot(t, raw_dof_vel[:, 4], t, filtered_dof_vel[:, 4])

    plt.subplot(816)
    plt.plot(t, raw_dof_vel[:, 10], t, filtered_dof_vel[:, 10])

    plt.subplot(817)
    plt.plot(t, raw_dof_vel[:, 11], t, filtered_dof_vel[:, 11])
    # plt.plot(t, raw_dof_vel[:, 11], t, filtered_base_quat[:, 11])

    plt.subplot(818)
    plt.plot(t, raw_dof_vel[:, 12], t, filtered_dof_vel[:, 12])

    fig_dof_vel_right = plt.figure()
    plt.subplot(811)
    plt.plot(t, raw_dof_vel[:, 5], t, filtered_dof_vel[:, 5])
    plt.title('dof_vel_right')

    plt.subplot(812)
    plt.plot(t, raw_dof_vel[:, 6], t, filtered_dof_vel[:, 6])

    plt.subplot(813)
    plt.plot(t, raw_dof_vel[:, 7], t, filtered_dof_vel[:, 7])

    plt.subplot(814)
    plt.plot(t, raw_dof_vel[:, 8], t, filtered_dof_vel[:, 8])

    plt.subplot(815)
    plt.plot(t, raw_dof_vel[:, 9], t, filtered_dof_vel[:, 9])

    plt.subplot(816)
    plt.plot(t, raw_dof_vel[:, 13], t, filtered_dof_vel[:, 13])

    plt.subplot(817)
    plt.plot(t, raw_dof_vel[:, 14], t, filtered_dof_vel[:, 14])

    plt.subplot(818)
    plt.plot(t, raw_dof_vel[:, 15], t, filtered_dof_vel[:, 15])


def plot_end_effector_pos(raw_end_effector_pos, filtered_dof_pos, t):
    fig_end_effector_pos_left = plt.figure()
    plt.subplot(611)
    plt.plot(t, raw_end_effector_pos[:, 0], t, filtered_end_effector_pos[:, 0])
    plt.title('end_effector_pos_left')

    plt.subplot(612)
    plt.plot(t, raw_end_effector_pos[:, 1], t, filtered_end_effector_pos[:, 1])

    plt.subplot(613)
    plt.plot(t, raw_end_effector_pos[:, 2], t, filtered_end_effector_pos[:, 2])

    plt.subplot(614)
    plt.plot(t, raw_end_effector_pos[:, 6], t, filtered_end_effector_pos[:, 6])

    plt.subplot(615)
    plt.plot(t, raw_end_effector_pos[:, 7], t, filtered_end_effector_pos[:, 7])

    plt.subplot(616)
    plt.plot(t, raw_end_effector_pos[:, 8], t, filtered_end_effector_pos[:, 8])

    fig_end_effector_pos_right = plt.figure()
    plt.subplot(611)
    plt.plot(t, raw_end_effector_pos[:, 3], t, filtered_end_effector_pos[:, 3])
    plt.title('end_effector_pos_right')

    plt.subplot(612)
    plt.plot(t, raw_end_effector_pos[:, 4], t, filtered_end_effector_pos[:, 4])

    plt.subplot(613)
    plt.plot(t, raw_end_effector_pos[:, 5], t, filtered_end_effector_pos[:, 5])

    plt.subplot(614)
    plt.plot(t, raw_end_effector_pos[:, 9], t, filtered_end_effector_pos[:, 9])

    plt.subplot(615)
    plt.plot(t, raw_end_effector_pos[:, 10],
             t, filtered_end_effector_pos[:, 10])

    plt.subplot(616)
    plt.plot(t, raw_end_effector_pos[:, 11],
             t, filtered_end_effector_pos[:, 11])


def plot_end_effector_vel(raw_end_effector_vel, filtered_dof_pos, t):
    fig_end_effector_vel_left = plt.figure()
    plt.subplot(611)
    plt.plot(t, raw_end_effector_vel[:, 0], t, filtered_end_effector_vel[:, 0])
    plt.title('end_effector_vel_left')

    plt.subplot(612)
    plt.plot(t, raw_end_effector_vel[:, 1], t, filtered_end_effector_vel[:, 1])

    plt.subplot(613)
    plt.plot(t, raw_end_effector_vel[:, 2], t, filtered_end_effector_vel[:, 2])

    plt.subplot(614)
    plt.plot(t, raw_end_effector_vel[:, 6], t, filtered_end_effector_vel[:, 6])

    plt.subplot(615)
    plt.plot(t, raw_end_effector_vel[:, 7], t, filtered_end_effector_vel[:, 7])

    plt.subplot(616)
    plt.plot(t, raw_end_effector_vel[:, 8], t, filtered_end_effector_vel[:, 8])

    fig_end_effector_vel_right = plt.figure()
    plt.subplot(611)
    plt.plot(t, raw_end_effector_vel[:, 3], t, filtered_end_effector_vel[:, 3])
    plt.title('end_effector_vel_right')

    plt.subplot(612)
    plt.plot(t, raw_end_effector_vel[:, 4], t, filtered_end_effector_vel[:, 4])

    plt.subplot(613)
    plt.plot(t, raw_end_effector_vel[:, 5], t, filtered_end_effector_vel[:, 5])

    plt.subplot(614)
    plt.plot(t, raw_end_effector_vel[:, 9], t, filtered_end_effector_vel[:, 9])

    plt.subplot(615)
    plt.plot(t, raw_end_effector_vel[:, 10],
             t, filtered_end_effector_vel[:, 10])

    plt.subplot(616)
    plt.plot(t, raw_end_effector_vel[:, 11],
             t, filtered_end_effector_vel[:, 11])


# -----------------------------------plot funciton----------------------------------- #


if __name__ == "__main__":

    main()
