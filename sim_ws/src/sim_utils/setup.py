from setuptools import setup

package_name = 'sim_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Matthew Bergman',
    maintainer_email='bergman9@purdue.edu',
    description='Package for supplementary nodes for ME597 simulation-based tasks',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_ball_controller = sim_utils.red_ball_control:main'
        ],
    },
)
