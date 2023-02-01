# General tools for python
import numpy as np
import itertools
from copy import deepcopy
from typing import Tuple

# Plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from scipy.spatial import transform
import gtsam



class Agent:
    def __init__(self, xbegin, xend, ybegin, yend, zbegin, zend, speed, is_2dim):
        self.xbegin = xbegin
        self.xend = xend
        self.ybegin = ybegin
        self.yend = yend
        self.zbegin = zbegin
        self.zend = zend

        self.two_dim = is_2dim
        self.speed = speed
        self.pos = None
        self.ori = None
        self.goal = None
    
    def get_random_pos(self):
        spot = np.zeros(3)
        spot[0] = np.random.uniform(self.xbegin, self.xend)
        spot[1] = np.random.uniform(self.ybegin, self.yend)
        if not self.two_dim:
            spot[2] = np.random.uniform(self.zbegin, self.zend)
        return spot
    
    def set_pose(self, initial_pos = None):
        if initial_pos is None:
            self.pos = self.get_random_pos()
        else:
            self.pos = initial_pos
        
        self.ori = np.array([0,0,0,1])

    def make_step(self):
        if self.goal is None:
            self.goal = self.get_random_pos()
            diff = self.goal - self.pos
            self.ori = np.zeros(4)
            self.ori[0:3] = np.cross(np.array([1,0,0]), diff)
            self.ori[3] = np.linalg.norm(diff) + diff[0]
            self.ori = self.ori / np.linalg.norm(self.ori)

        self.dir = self.goal - self.pos
        fract = self.speed / np.linalg.norm(self.dir)
        if fract >= 1.0:
            self.pos = self.goal
            self.goal = None
        else:
            self.pos += fract * self.dir

        return deepcopy(self.pos), deepcopy(self.ori)

#********************************************************

def organize_movement(num_agents, num_timesteps, speed, dims, only_2d_allowed):
    agents = []
    for n in range(num_agents):
        next_agent = Agent(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], speed, only_2d_allowed)
        next_agent.set_pose()
        agents.append(next_agent)
    
    poses = [ [] for _ in range(num_agents) ]
    # set initial position
    for i, agent in enumerate(agents):
            poses[i].append(np.concatenate((agent.pos, agent.ori)))
    for t in range(num_timesteps-1):
        for i, agent in enumerate(agents):
            new_pos, new_ori = agent.make_step()
            poses[i].append(np.concatenate((new_pos, new_ori)))
    
    return np.array(poses)

def add_noise(data, tnoise, rnoise):
    noisy_data = deepcopy(data)
    for meas in noisy_data:
        meas[-1][0:3] = np.random.multivariate_normal(meas[-1][0:3], tnoise)
        euler_angles = transform.Rotation.from_quat(meas[-1][3:7]).as_euler("xyz")
        noisy_euler_angles = np.random.multivariate_normal(euler_angles, rnoise)
        meas[-1][3:7] = transform.Rotation.from_euler("xyz", noisy_euler_angles).as_quat()
    return noisy_data

def get_pose_difference(pose1, pose2):
    ori1 = transform.Rotation.from_quat(pose1[3:7])
    ori2 = transform.Rotation.from_quat(pose2[3:7])
    ori12 = ori1.inv() * ori2
    transl = ori1.inv().apply(pose2[0:3] - pose1[0:3])
    rot = ori12.as_quat()
    return np.concatenate((transl, rot))


def get_measurements(poses, communication_distance):
    nagents, ntimesteps, _ = np.shape(poses)

    intra_measurements = []
    for a in range(nagents):
        for t in range(ntimesteps-1):
            specifier_pose1 = gtsam.symbol(robot_names[a], t)
            specifier_pose2 = gtsam.symbol(robot_names[a], t+1)
            intra_measurements.append([specifier_pose1, specifier_pose2, get_pose_difference(poses[a,t], poses[a,t+1])])

    inter_measurements = []
    for t in range(ntimesteps-1):
        for pose_comb in itertools.combinations(enumerate(poses[:,t]), 2):
            # pose_comb contains all possible combinations of different agents with the related index.
            # So for example for [x,y,z] it returns [((0,x),(1,y)), ((0,x),(2,z)), ((1,y),(2,z))]
            dist = np.linalg.norm(pose_comb[0][1][0:3] - pose_comb[1][1][0:3])
            if dist > communication_distance:
                continue
            specifier_pose1 = gtsam.symbol(robot_names[pose_comb[0][0]], t)
            specifier_pose2 = gtsam.symbol(robot_names[pose_comb[1][0]], t)
            inter_measurements.append([specifier_pose1, specifier_pose2, get_pose_difference(pose_comb[0][1], pose_comb[1][1])])

    return intra_measurements, inter_measurements

def write_to_file(estimations, intra_measurements, inter_measurements, filepath):
    nagents = len(estimations)
    ntimesteps = len(estimations[0])

    files = []
    for a in range(nagents):
        files.append(open(filepath + str(a) + ".g2o", "w"))
    files.append(open(filepath + "full_graph.g2o", "w"))

    for a in range(nagents):
        for t in range(ntimesteps):
            files[a].write("VERTEX_SE3:QUAT ")
            files[a].write(" " + str(gtsam.symbol(robot_names[a], t)))
            for num in estimations[a,t]:
                files[a].write(" %.6f" % num)
            files[a].write("\n")

            files[-1].write("VERTEX_SE3:QUAT")
            files[-1].write(" " + str(a * ntimesteps + t))
            for num in estimations[a,t]:
                files[-1].write(" %.6f" % num)
            files[-1].write("\n")

    for m in (intra_measurements + inter_measurements):
        a1 = robot_names.index(gtsam.Symbol(m[0]).string()[0])
        a2 = robot_names.index(gtsam.Symbol(m[1]).string()[0])
        t1 = gtsam.Symbol(m[0]).index()
        t2 = gtsam.Symbol(m[1]).index()

        files[-1].write("EDGE_SE3:QUAT")
        files[-1].write(" " + str(a1 * ntimesteps + t1))
        files[-1].write(" " + str(a2 * ntimesteps + t2))
        for num in m[2]:
            files[-1].write(" %.6f" % num)
        files[-1].write(" 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1")
        files[-1].write("\n")

        files[a1].write("EDGE_SE3:QUAT")
        files[a1].write(" " + str(m[0]))
        files[a1].write(" " + str(m[1]))
        for num in m[2]:
            files[a1].write(" %.6f" % num)
        files[a1].write(" 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1")
        files[a1].write("\n")

        if(a1 == a2):
            continue

        files[a2].write("EDGE_SE3:QUAT")
        files[a2].write(" " + str(m[0]))
        files[a2].write(" " + str(m[1]))
        for num in m[2]:
            files[a2].write(" %.6f" % num)
        files[a2].write(" 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1")
        files[a2].write("\n")

    for file in files:
        file.close()

def estimate_from_odometry(measurements, initial_poses):
    estimations = []
    for initial_pose in initial_poses:
        estimations.append([initial_pose])
    for m in measurements:
        agent = gtsam.Symbol(m[0]).string()[0]
        agent_index = robot_names.index(agent)
        old_pose = estimations[agent_index][-1]
        old_ori = transform.Rotation.from_quat(old_pose[3:7])
        new_ori = old_ori * transform.Rotation.from_quat(m[2][3:7])
        new_pos = old_pose[0:3] + old_ori.apply(m[2][0:3])
        estimations[agent_index].append(np.concatenate((new_pos, new_ori.as_quat())))
    return np.array(estimations)

def update(num, poses, lines, ax, plot_arrows):
    for i, line in enumerate(lines):
        all_poses_for_rob_i = poses[i, 0:num+1, 0:2]
        line.set_data(all_poses_for_rob_i[:,0], all_poses_for_rob_i[:,1])
        line.set_3d_properties(poses[i, 0:num+1, 2])
    
    if plot_arrows:
        for agent in poses:
            base = agent[num,0:3]
            rot_mat = transform.Rotation.from_quat(agent[num,3:7])
            head = rot_mat.apply(np.array([0.2,0,0])) + base
            ax.plot([base[0], head[0]], [base[1], head[1]], [base[2], head[2]], color='black', linestyle='solid')


def animate(data, ntimesteps):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 10.0])
    ax.set_zlabel('Z')

    lines_for_plotting = []
    for i in range(len(data)):
        line, = ax.plot(data[i,0:2,0], data[i,0:2,1], data[i,0:2,2])
        lines_for_plotting.append(line)

    ani = animation.FuncAnimation(fig, update, ntimesteps-1, fargs=(data, lines_for_plotting, ax, plot_arrows), \
        interval=10000/(ntimesteps-1), blit=False, repeat=False)
    plt.show()
    


#*********** USER-DEFINED VARIABLES *********************************************
robot_names = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

timesteps = 16
robots = 4
space_dimensions = [-5, 5, -5, 5, 0, 10]
plot_arrows = True

distance_for_inter_measurements = 2.0
# t_noise = 0.01 * np.eye(3)
# r_noise = 0.01 * np.eye(3)
t_noise = 0.01 * np.array([[1,0,0], [0,1,0],[0,0,0]])
r_noise = 0.01 * np.array([[0,0,0], [0,0,0],[0,0,1]])
robot_speed = 1.0
only_2d_allowed = True

path_to_ground_truth = "/home/leo/Desktop/project/gt_"
path_to_noisy_data = "/home/leo/Desktop/project/"


#********************************************************
poses = organize_movement(robots, timesteps, robot_speed, space_dimensions, only_2d_allowed)
intra_measurements, inter_measurements = get_measurements(poses, distance_for_inter_measurements)

noisy_odometry = add_noise(intra_measurements, t_noise, r_noise)
estimations = estimate_from_odometry(noisy_odometry, poses[:,0,:])

noisy_intra_measurements = add_noise(intra_measurements, t_noise, r_noise)
noisy_inter_measurements = add_noise(inter_measurements, t_noise, r_noise)

write_to_file(poses, intra_measurements, inter_measurements, path_to_ground_truth)
write_to_file(estimations, noisy_intra_measurements, noisy_inter_measurements, path_to_noisy_data)

animate(poses, timesteps)
animate(estimations,timesteps)


