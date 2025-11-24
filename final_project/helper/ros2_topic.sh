#!/bin/bash

# List all ROS2 topics
topics=$(ros2 topic list)

# Iterate through each topic and show info
for t in $topics; do
    echo "==============================="
    echo "Topic: $t"
    ros2 topic info "$t"
    echo
done
