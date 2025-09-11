// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from task_2_interfaces:msg/JointData.idl
// generated code does not contain a copyright notice

#ifndef TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__TRAITS_HPP_
#define TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "task_2_interfaces/msg/detail/joint_data__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'center'
#include "geometry_msgs/msg/detail/point32__traits.hpp"

namespace task_2_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointData & msg,
  std::ostream & out)
{
  out << "{";
  // member: center
  {
    out << "center: ";
    to_flow_style_yaml(msg.center, out);
    out << ", ";
  }

  // member: vel
  {
    out << "vel: ";
    rosidl_generator_traits::value_to_yaml(msg.vel, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const JointData & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: center
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "center:\n";
    to_block_style_yaml(msg.center, out, indentation + 2);
  }

  // member: vel
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel: ";
    rosidl_generator_traits::value_to_yaml(msg.vel, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const JointData & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace task_2_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use task_2_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const task_2_interfaces::msg::JointData & msg,
  std::ostream & out, size_t indentation = 0)
{
  task_2_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use task_2_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const task_2_interfaces::msg::JointData & msg)
{
  return task_2_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<task_2_interfaces::msg::JointData>()
{
  return "task_2_interfaces::msg::JointData";
}

template<>
inline const char * name<task_2_interfaces::msg::JointData>()
{
  return "task_2_interfaces/msg/JointData";
}

template<>
struct has_fixed_size<task_2_interfaces::msg::JointData>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Point32>::value> {};

template<>
struct has_bounded_size<task_2_interfaces::msg::JointData>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Point32>::value> {};

template<>
struct is_message<task_2_interfaces::msg::JointData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__TRAITS_HPP_
