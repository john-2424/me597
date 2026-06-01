# me597
ME59700AS Lab Repo

Project Archives: https://drive.google.com/drive/folders/12VroA-d8rddguSaayqEc3N36t1FRXnAG?usp=drive_link

## Repository Overview

This repository collects ME59700AS autonomous systems coursework built around ROS 2, Gazebo, TurtleBot/TurtleBot3 simulation, motion control, mapping, navigation, and perception. The main focus is the Fall 2025 final project, followed by supporting lab work and archived submissions.

The overview below is intentionally high level: it explains what each workspace is for, where the important code lives, and how the pieces fit together without turning the README into a lab manual.

## Main Focus: Fall 2025 Final Project

**Path:** [`final_project/sim_ws_Fall2025`](final_project/sim_ws_Fall2025)

The final project is a ROS 2 TurtleBot3 Gazebo autonomy stack for mapping, navigation, obstacle avoidance, and object localization. It aligns with the project tasks from the lecture slides and the submitted final presentation: autonomous mapping, static-obstacle navigation, bonus local replanning, and colored-object search/localization.

### What It Solves

1. **Task 1 - Autonomous Mapping**
   - Implements frontier-based exploration over an occupancy grid.
   - Uses `/map` and `/map_updates`, TF pose lookup, frontier detection/clustering, A* planning, path smoothing, and PID-based path following.
   - Includes safety behavior using LaserScan sectors and occupancy-grid checks so the robot can back up, turn away, replan, and stop when exploration is complete.
   - Publishes visualization data such as `/global_plan` and frontier markers for RViz.

2. **Task 2 - Navigation With Static Obstacles**
   - Navigates to RViz goals using the map generated from Task 1.
   - Combines AMCL pose, map/PGM data, LaserScan data, inflated static obstacles, and dynamic obstacle marking.
   - Uses global A* planning, blockage detection, pure-pursuit-style path following, and reactive fallback maneuvers when obstacles are too close.
   - Publishes `/global_plan`, `/local_plan`, `/cmd_vel`, and navigation timing.

3. **Task 2 Bonus - RRT* Local Replanning**
   - [`task2_bonus.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2_bonus.py) extends the navigation stack with RRT* detours.
   - When a future segment of the active path becomes blocked, the node searches for a safe local reconnect point and splices a local RRT* path back into the global route.

4. **Task 3 - Search and Localize Objects**
   - Builds an autonomous search pipeline for red, green, and blue balls.
   - Generates coverage waypoints from free space, orders them to reduce backtracking, navigates with A* plus RRT* detours, and performs 360-degree scans at waypoints.
   - Detects colored balls with HSV segmentation, shape/size filtering, multi-frame stability checks, and physics-aware rejection of false positives.
   - Fuses camera bearing, LiDAR range, and AMCL pose to publish world-frame object estimates through RViz markers, `PoseArray`, and `/red_pos`, `/green_pos`, `/blue_pos`.

### Final Project Structure

| Path | Purpose |
| --- | --- |
| [`final_project/sim_ws_Fall2025/README.md`](final_project/sim_ws_Fall2025/README.md) | Build and setup notes for the final project simulation workspace. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo) | Main Gazebo/ROS package with worlds, models, maps, parameters, launch files, and task scripts. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task1.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task1.py) | Autonomous frontier exploration and mapping. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2.py) | Map-based navigation with obstacle handling. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2_bonus.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task2_bonus.py) | RRT* local replanning variant for the Task 2 bonus. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task3.py`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4/task3.py) | Full search, navigation, perception, and localization pipeline. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/launch`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/launch) | Launch files for the house world, AMCL, SLAM mapping, map loading, navigation, and object spawning. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/models`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/models) | TurtleBot3, house, trash can, cricket ball, and colored-object Gazebo models. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/worlds`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/worlds) | Gazebo worlds for house, closed-house, empty-world, and bonus environments. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/maps`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/maps) | Saved map artifacts used by localization and navigation. |
| [`final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/params`](final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/params) | AMCL, map server, and SLAM Toolbox parameter files. |
| [`final_project/sim_ws_Fall2025/src/sim_utils`](final_project/sim_ws_Fall2025/src/sim_utils) | Supplemental simulation utilities, including the red ball controller node. |
| [`final_project/helper`](final_project/helper) | Helper map files, topic notes, and setup/reference material used alongside the final project. |

### Final Project Build Notes

The nested final-project README gives the original workspace setup. In short:

```bash
cd final_project/sim_ws_Fall2025
colcon build --symlink-install
source install/local_setup.bash
export TURTLEBOT3_MODEL=waffle
```

Common supporting packages include TurtleBot3 teleoperation, SLAM Toolbox, Navigation2, Gazebo ROS packages, OpenCV/cv_bridge, NumPy, PyYAML, and `pynput`.

Useful simulation launch entry points include:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_house_norviz.launch.py
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
ros2 launch turtlebot3_gazebo task_6.launch.py
```

The project archive link at the top of this README contains the larger final-project deliverables, including presentation and video artifacts that are not ideal to keep directly in the repository.

## Other Workspaces and Coursework

### `lab1_ws` - ROS 2 Fundamentals

[`lab1_ws`](lab1_ws) contains the introductory ROS 2 publisher/subscriber and service/client work.

- [`task_1`](lab1_ws/src/task_1) implements a basic talker/listener pair using `std_msgs/Float64`.
- [`task_2`](lab1_ws/src/task_2) extends the pattern to custom message and service usage.
- [`task_2_interfaces`](lab1_ws/src/task_2_interfaces) defines:
  - `JointData.msg` with a `geometry_msgs/Point32 center` and `float32 vel`.
  - `JointState.srv` with `x`, `y`, `z` request fields and a `valid` response.

### `lab2_ws` - PID Control With LaserScan Feedback

[`lab2_ws`](lab2_ws) contains [`task_3`](lab2_ws/src/task_3), a TurtleBot4 PID speed-control exercise.

- Reads front-distance measurements from `/scan`.
- Publishes velocity commands to `/robot/cmd_vel`.
- Includes PID tuning logic, launch support, and simple plotting/visualization utilities for analyzing controller behavior.

### `lab3_ws` - Mapping, A*, and Path Following

[`lab3_ws`](lab3_ws) contains [`task_4`](lab3_ws/src/task_4), a navigation workspace that connects map processing, A* planning, waypoint selection, and PID-based path following.

- Includes map assets (`.pgm`/`.yaml`), RViz config, launch files, and a TurtleBot4 navigator launch.
- [`auto_navigator.py`](lab3_ws/src/task_4/task_4/auto_navigator.py) ties together AMCL pose, RViz goals, A* planning, global path publication, and `/cmd_vel` control.
- Helper modules cover map preprocessing, A* data structures, waypoint tracking, and PID/PIDStar tuning.

### `lab4_ws` - Perception and Red-Ball Tracking

[`lab4_ws`](lab4_ws) contains two perception/simulation tasks.

- [`task_5`](lab4_ws/src/task_5) publishes image data from a provided video and detects visual targets using OpenCV/HSV processing. It publishes bounding boxes through `vision_msgs`.
- [`task_6`](lab4_ws/src/task_6) tracks a red ball in simulation using camera detection, LaserScan-based safety, PID control, and a search-state structure for reacquiring the target.

### `sim_ws` - Base TurtleBot3 Simulation Workspace

[`sim_ws`](sim_ws) is the base ME597 TurtleBot3 Gazebo workspace used by earlier simulation labs.

- Includes the TurtleBot3 Gazebo package, house worlds, object models, maps, RViz config, launch files, and navigation/SLAM parameters.
- Contains lab-oriented scripts for obstacle spawning and task behavior under `sim_ws/src/turtlebot3_gazebo/src/lab4`.
- Includes [`sim_utils`](sim_ws/src/sim_utils), a small Python package with supplemental simulation nodes.

### `vocv_ws` - Vision/OpenCV ROS Packages

[`vocv_ws`](vocv_ws) contains ROS vision support packages from `vision_opencv`.

- [`cv_bridge`](vocv_ws/src/vision_opencv/cv_bridge) converts between ROS image messages and OpenCV image data.
- [`image_geometry`](vocv_ws/src/vision_opencv/image_geometry) provides camera model utilities.
- [`opencv_tests`](vocv_ws/src/vision_opencv/opencv_tests) contains example/test nodes and support files.

### `_submissions` - Archived Lab Submissions

[`_submissions`](_submissions) stores versioned zip files and expanded submission snapshots for labs 1-4.

It includes packaged lab archives, expanded task folders, maps, ROS bag metadata/data, and extra-credit artifacts. This directory is useful for preserving what was submitted at each stage, while the main lab workspaces remain easier to browse as source.

## Top-Level Files

| Path | Purpose |
| --- | --- |
| [`.gitignore`](.gitignore) | Ignore rules for Python, ROS, build products, editor files, and generated artifacts. |
| [`LICENSE.txt`](LICENSE.txt) | Repository license text. |
| [`README.md`](README.md) | This overview. |
| [`.vscode`](.vscode) | Local editor configuration. |

## Suggested Reading Order

1. Start with the final project task scripts in `final_project/sim_ws_Fall2025/src/turtlebot3_gazebo/src/lab4`.
2. Read `final_project/sim_ws_Fall2025/README.md` for environment setup.
3. Review `lab3_ws` if you want the earlier A*/waypoint-navigation foundation.
4. Review `lab4_ws` for the perception and red-ball tracking pieces that support the final project's object-search direction.
5. Use `_submissions` only when you need historical submitted versions or packaged artifacts.
