from quadrotor_mpc_lemniscate import QuadrotorMPC
from quadrotor2 import Quadrotor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import product

from math import sqrt

import numpy as np

# Animate the quadrotor trajectory
def plot_quadrotor_trajectory(quad, mpc_solver, x_sim, u_sim, save_anim=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')  

    trajectory = ax.plot([], [], [])[0]
    quadrotor_frame = [ax.plot([], [], [], color='black')[0], ax.plot([], [], [],color='black')[0]]

    num_steps = x_sim.shape[0]

    ax.set(xlim3d=(-4, 4), xlabel='X')
    ax.set(ylim3d=(-4, 4), ylabel='Y')
    ax.set(zlim3d=(-4, 4), zlabel='Z')

    anim = animation.FuncAnimation(
            fig, update_quad_trajectory, num_steps, 
            fargs=(quad, x_sim, u_sim, trajectory, quadrotor_frame), interval=round(500*(mpc_solver.horizon/mpc_solver.num_steps)))

    if save_anim:
        anim.save('quadrotor_mpc_test1.gif', writer='imagemagick', fps=(mpc_solver.num_steps/mpc_solver.horizon))
    plt.show()

# Plotting function used to construct quadrotor trajectory animation
def update_quad_trajectory(traj_index, quad, x_sim, u_sim, trajectory, quadrotor_frame):
    trajectory.set_data(x_sim[:traj_index+1, :2].T)
    trajectory.set_3d_properties(x_sim[:traj_index+1, 2])
    Rotm = np.transpose(quad.euler_to_rotm(x_sim[traj_index,6:9]))
    arm_length_scaling = 4.
    quadrotor_frame_motor1 = x_sim[traj_index,:3] \
                            + Rotm @ (sqrt(2.) / 2. * quad.arm_length * arm_length_scaling * np.array([1., 1., 0.]))
    quadrotor_frame_motor2 = x_sim[traj_index,:3] \
                            + Rotm @ (sqrt(2.) / 2. * quad.arm_length * arm_length_scaling * np.array([1., -1., 0.]))
    quadrotor_frame_motor3 = x_sim[traj_index,:3] \
                            + Rotm @ (sqrt(2.) / 2. * quad.arm_length * arm_length_scaling * np.array([-1., -1., 0.]))
    quadrotor_frame_motor4 = x_sim[traj_index,:3] \
                            + Rotm @ (sqrt(2.) / 2. * quad.arm_length * arm_length_scaling * np.array([-1., 1., 0.]))
    quadrotor_frame_13 = np.vstack((quadrotor_frame_motor1,quadrotor_frame_motor3))
    quadrotor_frame_24 = np.vstack((quadrotor_frame_motor2,quadrotor_frame_motor4))
    quadrotor_frame[0].set_data(quadrotor_frame_13[:,:2].T)
    quadrotor_frame[0].set_3d_properties(quadrotor_frame_13[:,2])
    quadrotor_frame[1].set_data(quadrotor_frame_24[:,:2].T)
    quadrotor_frame[1].set_3d_properties(quadrotor_frame_24[:,2])

        
def mpc_callback(status, x_mpc, u_mpc):
    return status, x_mpc, u_mpc

if __name__ == '__main__':
    horizon = 3.0
    quadrotor = Quadrotor(mass=0.027,arm_length=0.044, Ixx=2.3951e-5, Iyy=2.3951e-5, Izz=3.2347e-5, cm=2.4e-6, tau=0.1)
    x0 = np.array([0., 0., 1.0, 0., 0., 0., 0., 0., 0., 0.])
    xr = np.array([2., 1., 1.0, 0., 0., 0., 0., 0., 0., horizon])
    generate_c_code = True
    mpc_solver = QuadrotorMPC("crazyflie", generate_c_code, quadrotor, horizon, 20, None, x0=x0, xr=xr)
    status, x_mpc, u_mpc = mpc_solver.solve_mpc_trajectory(x0=x0)
    # np.save('x_mpc', x_mpc)
    plot_quadrotor_trajectory(quadrotor, mpc_solver, x_mpc, u_mpc, save_anim=False)
