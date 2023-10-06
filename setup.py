import os
from glob import glob
from setuptools import setup

package_name = 'crazyflie_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Llanes',
    maintainer_email='christian.llanes@gatech.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crazyflie_multiagent_mpc = crazyflie_mpc.crazyflie_multiagent_mpc:main'
        ],
    },
)
