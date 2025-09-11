// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from task_2_interfaces:srv/JointState.idl
// generated code does not contain a copyright notice

#ifndef TASK_2_INTERFACES__SRV__DETAIL__JOINT_STATE__BUILDER_HPP_
#define TASK_2_INTERFACES__SRV__DETAIL__JOINT_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "task_2_interfaces/srv/detail/joint_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace task_2_interfaces
{

namespace srv
{

namespace builder
{

class Init_JointState_Request_z
{
public:
  explicit Init_JointState_Request_z(::task_2_interfaces::srv::JointState_Request & msg)
  : msg_(msg)
  {}
  ::task_2_interfaces::srv::JointState_Request z(::task_2_interfaces::srv::JointState_Request::_z_type arg)
  {
    msg_.z = std::move(arg);
    return std::move(msg_);
  }

private:
  ::task_2_interfaces::srv::JointState_Request msg_;
};

class Init_JointState_Request_y
{
public:
  explicit Init_JointState_Request_y(::task_2_interfaces::srv::JointState_Request & msg)
  : msg_(msg)
  {}
  Init_JointState_Request_z y(::task_2_interfaces::srv::JointState_Request::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_JointState_Request_z(msg_);
  }

private:
  ::task_2_interfaces::srv::JointState_Request msg_;
};

class Init_JointState_Request_x
{
public:
  Init_JointState_Request_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointState_Request_y x(::task_2_interfaces::srv::JointState_Request::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_JointState_Request_y(msg_);
  }

private:
  ::task_2_interfaces::srv::JointState_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::task_2_interfaces::srv::JointState_Request>()
{
  return task_2_interfaces::srv::builder::Init_JointState_Request_x();
}

}  // namespace task_2_interfaces


namespace task_2_interfaces
{

namespace srv
{

namespace builder
{

class Init_JointState_Response_valid
{
public:
  Init_JointState_Response_valid()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::task_2_interfaces::srv::JointState_Response valid(::task_2_interfaces::srv::JointState_Response::_valid_type arg)
  {
    msg_.valid = std::move(arg);
    return std::move(msg_);
  }

private:
  ::task_2_interfaces::srv::JointState_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::task_2_interfaces::srv::JointState_Response>()
{
  return task_2_interfaces::srv::builder::Init_JointState_Response_valid();
}

}  // namespace task_2_interfaces

#endif  // TASK_2_INTERFACES__SRV__DETAIL__JOINT_STATE__BUILDER_HPP_
