// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from task_2_interfaces:msg/JointData.idl
// generated code does not contain a copyright notice

#ifndef TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__STRUCT_H_
#define TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'center'
#include "geometry_msgs/msg/detail/point32__struct.h"

/// Struct defined in msg/JointData in the package task_2_interfaces.
typedef struct task_2_interfaces__msg__JointData
{
  geometry_msgs__msg__Point32 center;
  float vel;
} task_2_interfaces__msg__JointData;

// Struct for a sequence of task_2_interfaces__msg__JointData.
typedef struct task_2_interfaces__msg__JointData__Sequence
{
  task_2_interfaces__msg__JointData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} task_2_interfaces__msg__JointData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__STRUCT_H_
