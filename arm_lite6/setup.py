from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'my_lite6_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('my_lite6_control/launch/*.py')),
        (os.path.join('share', package_name, 'scripts'), glob('my_lite6_control/scripts/*.sh')),
        (os.path.join('lib', package_name), glob('scripts_ros/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'custom_ik = my_lite6_control.custom_ik:main',
            'target = my_lite6_control.target:main',
        ],
    },
)
