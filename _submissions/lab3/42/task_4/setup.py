import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'task_4'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='srajule',
    maintainer_email='srajule@purdue.edu',
    description='Lab 3 Task 4 Navigation - Mapping with SLAM, Path Planning with A*, and Path Following (Navigation)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_navigator.py = task_4.auto_navigator:main'
        ],
    },
)
