import sys
import logging 
import numpy as np
import rowan

from crazyflie_py import *
import rclpy
import rclpy.node

from .quadrotor_simplified_model import Quadrotor
from .trajectory_tracking_mpc import QuadrotorMpcTrajectory

from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint

import pathlib

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

from time import time
from rclpy import executors
from rclpy.qos import qos_profile_sensor_data

import argparse

from ament_index_python.packages import get_package_share_directory

import tf_transformations

from enum import Enum

class Motors(Enum):
    MOTOR_CLASSIC = 1 # https://store.bitcraze.io/products/4-x-7-mm-dc-motor-pack-for-crazyflie-2 w/ standard props
    MOTOR_UPGRADE = 2 # https://store.bitcraze.io/collections/bundles/products/thrust-upgrade-bundle-for-crazyflie-2-x

class CrazyflieMPC(rclpy.node.Node):
    def __init__(self, node_name, mpc_N, mpc_horizon, rate):
        super().__init__(node_name)
        # super(CrazyflieMPC, self).__init__(target=self._start_agent)
        # self._swarm = swarm
        # self.time_helper = self._swarm.timeHelper
        # self.cfserver = self._swarm.allcfs
        # self._cf = self.cfserver.crazyfliesByName[name]
        name = self.get_name()
        prefix = '/' + name
        
        self.is_connected = True

        self.rate = rate

        self.odometry = Odometry()

        self.mpc_N = mpc_N
        self.mpc_horizon = mpc_horizon

        self.position = []
        self.velocity = []
        self.attitude = []

        self.trajectory_changed = True

        self.flight_mode = 'idle'
        self.trajectory_t0 = self.get_clock().now()
        self.trajectory_type = 'lemniscate'
        
        # TODO: Switch to parameters yaml?
        self.motors = Motors.MOTOR_UPGRADE # MOTOR_CLASSIC, MOTOR_UPGRADE

        self.takeoff_duration = 2.0
        self.land_duration = 2.0

        self.m = 0.028
        self.g = 9.80665

        quadrotor = Quadrotor(mass=self.m, arm_length=0.044, Ixx=2.3951e-5, Iyy=2.3951e-5, Izz=3.2347e-5, cm=2.4e-6, tau=0.08)
        self.acados_c_generated_code_path = pathlib.Path(get_package_share_directory('crazyflie_mpc')).resolve() / 'acados_generated_files'
        self.mpc_solver = QuadrotorMpcTrajectory('crazyflie', quadrotor, mpc_horizon, mpc_N, code_export_directory=self.acados_c_generated_code_path)
        self.get_logger().info('Initialization completed...')

        self.is_flying = False
        
        self.create_subscription(
            PoseStamped,
            f'{prefix}/pose',
            self._pose_msg_callback,
            10)
        
        self.create_subscription(
            LogDataGeneric,
            f'{prefix}/velocity',
            self._velocity_msg_callback,
            10)
        
        self.mpc_solution_path_pub = self.create_publisher(
            Path,
            f'{prefix}/mpc_solution_path',
            10)
        
        self.attitude_setpoint_pub = self.create_publisher(
            AttitudeSetpoint,
            f'{prefix}/cmd_attitude_setpoint',
            10)
        
        self.takeoffService = self.create_subscription(Empty, f'/all/mpc_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/mpc_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/mpc_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/mpc_hover', self.hover, 10)

        self.create_timer(1/rate, self._main_loop)

    def _pose_msg_callback(self, msg: PoseStamped):
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude = tf_transformations.euler_from_quaternion([msg.pose.orientation.x,
                                                                  msg.pose.orientation.y,
                                                                  msg.pose.orientation.z,
                                                                  msg.pose.orientation.w])

    def _velocity_msg_callback(self, msg: LogDataGeneric):
        self.velocity = msg.values

    def start_trajectory(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'trajectory'

    def takeoff(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'takeoff'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        1.0])
        
    def hover(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'hover'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        self.position[2]])

    def land(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'land'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        0.1])
            
    def trajectory_function(self, t):
        if self.trajectory_type == 'horizontal_circle':      
            a = 1.0
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.cos(omega*t) - a
            pyr = self.trajectory_start_position[1] + a*np.sin(omega*t)
            pzr = self.trajectory_start_position[2]
            vxr = -a*omega*np.sin(omega*t)
            vyr = a*omega*np.cos(omega*t)
            vzr = 0.0
        elif self.trajectory_type == 'vertical_circle':
            a = 1.0
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.sin(-omega*t + np.pi)
            pyr = self.trajectory_start_position[1]
            pzr = self.trajectory_start_position[2] + a*np.cos(-omega*t + np.pi) + a
            vxr = -a*omega*np.cos(-omega*t + np.pi)
            vyr = 0.0
            vzr = a*omega*np.sin(-omega*t + np.pi)
        elif self.trajectory_type == 'tilted_circle':
            a = 0.5
            c = 0.3
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.cos(omega*t) - a
            pyr = self.trajectory_start_position[1] + a*np.sin(omega*t)
            pzr = self.trajectory_start_position[2] + c*np.sin(omega*t)
            vxr = -a*omega*np.sin(omega*t)
            vyr = a*omega*np.cos(omega*t)
            vzr = c*omega*np.cos(omega*t)
        elif self.trajectory_type == 'lemniscate':
            a = 1.0
            b = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.sin(b*t)
            pyr = self.trajectory_start_position[1] + a*np.sin(b*t)*np.cos(b*t)
            pzr = self.trajectory_start_position[2]
            vxr = a*b*np.cos(b*t)
            vyr = a*b*np.cos(2*b*t)
            vzr = 0.0
        elif self.trajectory_type == 'helix':
            a = 1.0
            T_end = 10.0
            helix_velocity = 0.2
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.cos(omega*t) - a
            pyr = self.trajectory_start_position[1] + a*np.sin(omega*t)
            vxr = -a*omega*np.sin(omega*t)
            vyr = a*omega*np.cos(omega*t)
            if t < T_end:
                pzr = self.trajectory_start_position[2] + helix_velocity*t
                vzr = helix_velocity
            else:
                pzr = self.trajectory_start_position[2] + helix_velocity*T_end
                vzr = 0.0
        return np.array([pxr,pyr,pzr,vxr,vyr,vzr,0.,0.,0.,*np.array([0., 0., 0., self.g*self.m])])

    def navigator(self, t):
        hover_control = np.array([0., 0., 0., self.g*self.m])
        if self.flight_mode == 'takeoff':
            t_mpc_array = np.linspace(t, self.mpc_horizon + t, self.mpc_N)
            yref = np.array([np.array([*((self.go_to_position - self.trajectory_start_position)*(1./(1. + np.exp(-(12.0 * (t_mpc - self.takeoff_duration) / self.takeoff_duration + 6.0)))) + self.trajectory_start_position),0.,0.,0.,0.,0.,0.,*hover_control]) for t_mpc in t_mpc_array]).T
            # yref = np.repeat(np.array([[*self.go_to_position,0,0,0]]).T, self.mpc_N, axis=1)
        elif self.flight_mode == 'land':
            t_mpc_array = np.linspace(t, self.mpc_horizon + t, self.mpc_N)
            yref = np.array([np.array([*((self.go_to_position - self.trajectory_start_position)*(1./(1. + np.exp(-(12.0 * (t_mpc - self.land_duration) / self.land_duration + 6.0)))) + self.trajectory_start_position),0.,0.,0.,0.,0.,0.,*hover_control]) for t_mpc in t_mpc_array]).T
            # yref = np.repeat(np.array([[*self.go_to_position,0,0,0]]).T, self.mpc_N, axis=1)
        elif self.flight_mode == 'trajectory':
            t_mpc_array = np.linspace(t, self.mpc_horizon + t, self.mpc_N)
            yref = np.array([self.trajectory_function(t_mpc) for t_mpc in t_mpc_array]).T
        elif self.flight_mode == 'hover':
            yref = np.repeat(np.array([[*self.go_to_position,0.,0.,0.,0.,0.,0.,*hover_control]]).T, self.mpc_N, axis=1)
        return yref
    
    def cmd_attitude_setpoint(self, roll, pitch, yaw_rate, thrust_pwm):
        setpoint = AttitudeSetpoint()
        setpoint.roll = roll
        setpoint.pitch = pitch
        setpoint.yaw_rate = yaw_rate
        setpoint.thrust = thrust_pwm
        self.attitude_setpoint_pub.publish(setpoint)

    def thrust_to_pwm(self, collective_thrust: float) -> int:
        # omega_per_rotor = 7460.8*np.sqrt((collective_thrust / 4.0))
        # pwm_per_rotor = 24.5307*(omega_per_rotor - 380.8359)
        if self.motors == Motors.MOTOR_CLASSIC:
            return int(max(min(24.5307*(7460.8*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
        elif self.motors == Motors.MOTOR_UPGRADE:
            return int(max(min(24.5307*(6462.1*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))

    def _main_loop(self):
        if self.flight_mode == 'idle':
            return

        if not self.position or not self.velocity or not self.attitude:
            self.get_logger().warning("Empty state message.")
            return
        
        if not self.is_flying:
            self.is_flying = True
            self.cmd_attitude_setpoint(0.,0.,0.,0)

        if self.trajectory_changed:
            self.trajectory_start_position = self.position
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False

        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9

        rpy_rad = np.radians([self.attitude[0], -self.attitude[1], self.attitude[2]])

        x0 = np.array([
            *self.position,
            *self.velocity,
            *rpy_rad
        ])

        yref = self.navigator(t)
        status, x_mpc, u_mpc = self.mpc_solver.solve_mpc_trajectory(x0, yref)
        
        # thrust = int(min(1.77*u_mpc[0, 3] / 9.80665 * 1000 * 1000, 60000))
        thrust_pwm = self.thrust_to_pwm(u_mpc[0, 3])

        mpc_solution_path = Path()
        mpc_solution_path.header.frame_id = 'world'
        mpc_solution_path.header.stamp = self.get_clock().now().to_msg()

        for i in range(self.mpc_N):
            mpc_pose = PoseStamped()
            mpc_pose.pose.position.x = x_mpc[i,0]
            mpc_pose.pose.position.y = x_mpc[i,1]
            mpc_pose.pose.position.z = x_mpc[i,2]
            mpc_solution_path.poses.append(mpc_pose)

        self.mpc_solution_path_pub.publish(mpc_solution_path)

            
        self.cmd_attitude_setpoint(np.degrees(u_mpc[0,0]), 
                                    np.degrees(u_mpc[0,1]), 
                                    0.0, 
                                    thrust_pwm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--n_agents",help ="Number of agents",default=1,type=int)
    args = parser.parse_args()
    N_AGENTS = args.n_agents

    rclpy.init()
    mpc_tf = 3.0
    mpc_N = 20
    rate = 50
    nodes = [CrazyflieMPC('cf_'+str(i), mpc_N, mpc_tf, rate) for i in np.arange(1, 1 + N_AGENTS)]
    executor = executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        while rclpy.ok():
            node.get_logger().info('Beginning multiagent executor, shut down with CTRL-C')
            executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')

    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()
