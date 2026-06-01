# me597
ME59700AS Final Project Repo

Project Archives: https://drive.google.com/drive/folders/12VroA-d8rddguSaayqEc3N36t1FRXnAG?usp=drive_link

## Final Project: TurtleBot Autonomous Mapping, Navigation, and Object Localization

This repository presents the Fall 2025 ME59700AS final project: a ROS 2 TurtleBot3 Gazebo autonomy stack for autonomous mapping, map-based navigation, static-obstacle avoidance, local replanning, and colored-object search/localization.

The implementation is centered in [`final_project/sim_ws_Fall2025`](final_project/sim_ws_Fall2025). It uses occupancy grids, TF, AMCL, LaserScan data, A* planning, RRT* local replanning, PID control, OpenCV-based color detection, and RViz visualization.

## Project Tasks

### Task 1: Autonomous Mapping

[`task1.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task1.py) explores an unknown Gazebo environment and builds a map using frontier exploration.

Key ideas:

- Detects and clusters frontier cells from `/map` and `/map_updates`.
- Uses TF to locate the robot in the map frame.
- Selects safe exploration goals with obstacle clearance.
- Plans with grid-based A* and smooths the path where line-of-sight is safe.
- Follows paths with PID control and LaserScan-based safety checks.
- Stops cleanly when no useful frontiers remain.

### Task 2: Navigation With Static Obstacles

[`task2.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2.py) navigates to RViz goals using the map from Task 1 while avoiding static and sensed obstacles.

Key ideas:

- Uses AMCL pose, map data, LaserScan data, and RViz goal poses.
- Inflates static and dynamic obstacles to preserve robot clearance.
- Plans global routes with A*.
- Detects blocked path segments during execution.
- Uses pure-pursuit-style path following with speed reduction near turns and goals.
- Falls back to reactive avoidance when obstacles are too close.

### Task 2 Bonus: RRT* Local Replanning

[`task2_bonus.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2_bonus.py) extends the Task 2 navigation stack with local RRT* replanning.

When the active A* path becomes blocked, the node searches for a safe reconnect point, generates a local RRT* detour, publishes it as `/local_plan`, and splices it into the active route.

### Task 3: Object Search and Localization

[`task3.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task3.py) autonomously searches for red, green, and blue objects and estimates their world-frame locations.

Key ideas:

- Generates coverage waypoints from free-space structure.
- Navigates between waypoints using A* with RRT* detours.
- Performs 360-degree scans at search locations.
- Detects colored balls with HSV segmentation, shape filtering, and multi-frame stability checks.
- Combines camera bearing, LiDAR range, and AMCL pose to estimate object positions.
- Publishes RViz markers, `PoseArray`, and `/red_pos`, `/green_pos`, `/blue_pos`.

## Important Project Files

| Path | Purpose |
| --- | --- |
| [`final_project/sim_ws_Fall2025`](final_project/sim_ws_Fall2025) | Main ROS 2 final project workspace. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo) | Gazebo worlds, TurtleBot3 models, launch files, maps, params, and task scripts. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4) | Main task implementations. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/launch`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/launch) | Launch files for simulation, mapping, localization, navigation, and object spawning. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/maps`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/maps) | Saved maps used for localization and navigation. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/params`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/params) | AMCL, map server, and SLAM Toolbox configuration. |
| [`final_project/sim_ws_Fall2025/src/sim_utils`](final_project/sim_ws_Fall2025/src/sim_utils) | Supplemental simulation utilities. |
| [`final_project/helper`](final_project/helper) | Helper maps, ROS topic notes, and project support files. |

## Setup

Build and source the final project workspace:

```bash
cd final_project/sim_ws_Fall2025
colcon build --symlink-install
source install/local_setup.bash
export TURTLEBOT3_MODEL=waffle
```

Useful supporting packages include:

```bash
sudo apt install ros-humble-turtlebot3-teleop
sudo apt install ros-humble-slam-toolbox
sudo apt install ros-humble-navigation2
pip install pynput
```

Useful launch entry points:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_house_norviz.launch.py
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
ros2 launch turtlebot3_gazebo task_6.launch.py
```

## Project Artifacts

The archive link at the top of this README contains larger final-project artifacts such as presentation files, videos, maps, and submitted deliverables that are better kept outside the Git repository.
