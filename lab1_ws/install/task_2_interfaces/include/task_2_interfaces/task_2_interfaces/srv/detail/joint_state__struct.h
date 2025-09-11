// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from task_2_interfaces:srv/JointState.idl
// generated code does not contain a copyright notice

#ifndef TASK_2_INTERFACES__SRV__DETAIL__JOINT_STATE__STRUCT_H_
#define TASK_2_INTERFACES__SRV__DETAIL__JOINT_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/JointState in the package task_2_interfaces.
typedef struct task_2_interfaces__srv__JointState_Request
{
  float x;
  float y;
  float z;
} task_2_interfaces__srv__JointState_Request;

// Struct for a sequence of task_2_interfaces__srv__JointState_Request.
typedef struct task_2_interfaces__srv__JointState_Request__Sequence
{
  task_2_interfaces__srv__JointState_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} task_2_interfaces__srv__JointState_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/JointState in the package task_2_interfaces.
typedef struct task_2_interfaces__srv__JointState_Response
{
  bool valid;
} task_2_interfaces__srv__JointState_Response;

// Struct for a sequence of task_2_interfaces__srv__JointState_Response.
typedef struct task_2_interfaces__srv__JointState_Response__Sequence
{
  task_2_interfaces__srv__JointState_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} task_2_interfaces__srv__JointState_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TASK_2_INTERFACES__SRV__DETAIL__JOINT_STATE__STRUCT_H_
