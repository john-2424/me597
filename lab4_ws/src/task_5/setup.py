import os
from setuptools import find_packages, setup

package_name = 'task_5'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'Resources'), ['Resources/lab4_video.avi']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='srajule',
    maintainer_email='srajule@purdue.edu',
    description='Lab 4: Task 5: Perception: Camera',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher = task_5.image_publisher:main'
        ],
    },
)
