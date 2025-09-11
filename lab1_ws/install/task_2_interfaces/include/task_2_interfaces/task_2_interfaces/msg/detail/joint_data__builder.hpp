// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from task_2_interfaces:msg/JointData.idl
// generated code does not contain a copyright notice

#ifndef TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__BUILDER_HPP_
#define TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "task_2_interfaces/msg/detail/joint_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace task_2_interfaces
{

namespace msg
{

namespace builder
{

class Init_JointData_vel
{
public:
  explicit Init_JointData_vel(::task_2_interfaces::msg::JointData & msg)
  : msg_(msg)
  {}
  ::task_2_interfaces::msg::JointData vel(::task_2_interfaces::msg::JointData::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return std::move(msg_);
  }

private:
  ::task_2_interfaces::msg::JointData msg_;
};

class Init_JointData_center
{
public:
  Init_JointData_center()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointData_vel center(::task_2_interfaces::msg::JointData::_center_type arg)
  {
    msg_.center = std::move(arg);
    return Init_JointData_vel(msg_);
  }

private:
  ::task_2_interfaces::msg::JointData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::task_2_interfaces::msg::JointData>()
{
  return task_2_interfaces::msg::builder::Init_JointData_center();
}

}  // namespace task_2_interfaces

#endif  // TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__BUILDER_HPP_
