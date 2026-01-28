# ME597 sim_ws
A simulation workspace bulit for ME597 final project in Fall 2025. 

## Instructions:
1. Simply save this workspace e.g., 
    ```
    cd ~/ros2 # cd into the dir you want to keep this workspace in
    git clone https://github.com/naslab-projects/sim_ws_Fall2025.git
    ```

2. In a new terminal build the sim_ws workspace: 
    ```
    cd sim_ws_Fall2025
    colcon build --symlink-install
    ```

3. Add turtlebot3 environment variable to .bashrc file
    ```
    echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
    ```
4. Run these to install useful packages you will use in the simulator.
    ```
    sudo apt install ros-humble-turtlebot3-teleop
    sudo apt install ros-humble-slam-toolbox
    sudo apt install ros-humble-navigation2
    ```
    
    ```
    pip install pynput
    ```

5. Don't forget to source the workspace whenever you use it
    ```
    cd sim_ws
    source install/local_setup.bash
    ```
