from launch import LaunchDescription
from launch_ros.actions import Node
# import argparse



def generate_launch_description():
    return LaunchDescription([
    Node(
        package='crazyflie_mpc',
        executable='crazyflie_multiagent_mpc',
    ),
    ])