// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from task_2_interfaces:srv/JointState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "task_2_interfaces/srv/detail/joint_state__rosidl_typesupport_introspection_c.h"
#include "task_2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "task_2_interfaces/srv/detail/joint_state__functions.h"
#include "task_2_interfaces/srv/detail/joint_state__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  task_2_interfaces__srv__JointState_Request__init(message_memory);
}

void task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_fini_function(void * message_memory)
{
  task_2_interfaces__srv__JointState_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_member_array[3] = {
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(task_2_interfaces__srv__JointState_Request, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(task_2_interfaces__srv__JointState_Request, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "z",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(task_2_interfaces__srv__JointState_Request, z),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_members = {
  "task_2_interfaces__srv",  // message namespace
  "JointState_Request",  // message name
  3,  // number of fields
  sizeof(task_2_interfaces__srv__JointState_Request),
  task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_member_array,  // message members
  task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_type_support_handle = {
  0,
  &task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_task_2_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState_Request)() {
  if (!task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_type_support_handle.typesupport_identifier) {
    task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &task_2_interfaces__srv__JointState_Request__rosidl_typesupport_introspection_c__JointState_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "task_2_interfaces/srv/detail/joint_state__rosidl_typesupport_introspection_c.h"
// already included above
// #include "task_2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "task_2_interfaces/srv/detail/joint_state__functions.h"
// already included above
// #include "task_2_interfaces/srv/detail/joint_state__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  task_2_interfaces__srv__JointState_Response__init(message_memory);
}

void task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_fini_function(void * message_memory)
{
  task_2_interfaces__srv__JointState_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_member_array[1] = {
  {
    "valid",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(task_2_interfaces__srv__JointState_Response, valid),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_members = {
  "task_2_interfaces__srv",  // message namespace
  "JointState_Response",  // message name
  1,  // number of fields
  sizeof(task_2_interfaces__srv__JointState_Response),
  task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_member_array,  // message members
  task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_type_support_handle = {
  0,
  &task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_task_2_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState_Response)() {
  if (!task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_type_support_handle.typesupport_identifier) {
    task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &task_2_interfaces__srv__JointState_Response__rosidl_typesupport_introspection_c__JointState_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "task_2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "task_2_interfaces/srv/detail/joint_state__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_members = {
  "task_2_interfaces__srv",  // service namespace
  "JointState",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_Request_message_type_support_handle,
  NULL  // response message
  // task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_Response_message_type_support_handle
};

static rosidl_service_type_support_t task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_type_support_handle = {
  0,
  &task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_task_2_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState)() {
  if (!task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_type_support_handle.typesupport_identifier) {
    task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, task_2_interfaces, srv, JointState_Response)()->data;
  }

  return &task_2_interfaces__srv__detail__joint_state__rosidl_typesupport_introspection_c__JointState_service_type_support_handle;
}
