import os
import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # load crazyflies
    crazyflies_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'crazyswarm2_robots_sim.yaml')
    
    
    crazyswarm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('crazyflie'), 'launch'), '/launch.py']),
        launch_arguments={'crazyflies_yaml_file': crazyflies_yaml, 'backend': 'cflib', 'mocap': 'False', 'gui': 'False', 'rviz': 'True'}.items()
    )   

    crazyflie_mpc_node = Node(
        package='crazyflie_mpc',
        executable='crazyflie_multiagent_mpc',
    )

    ld =  LaunchDescription()
    ld.add_action(crazyswarm2_launch)
    ld.add_action(crazyflie_mpc_node)

    return ld