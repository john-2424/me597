// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from task_2_interfaces:msg/JointData.idl
// generated code does not contain a copyright notice

#ifndef TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__STRUCT_HPP_
#define TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'center'
#include "geometry_msgs/msg/detail/point32__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__task_2_interfaces__msg__JointData __attribute__((deprecated))
#else
# define DEPRECATED__task_2_interfaces__msg__JointData __declspec(deprecated)
#endif

namespace task_2_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct JointData_
{
  using Type = JointData_<ContainerAllocator>;

  explicit JointData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : center(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->vel = 0.0f;
    }
  }

  explicit JointData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : center(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->vel = 0.0f;
    }
  }

  // field types and members
  using _center_type =
    geometry_msgs::msg::Point32_<ContainerAllocator>;
  _center_type center;
  using _vel_type =
    float;
  _vel_type vel;

  // setters for named parameter idiom
  Type & set__center(
    const geometry_msgs::msg::Point32_<ContainerAllocator> & _arg)
  {
    this->center = _arg;
    return *this;
  }
  Type & set__vel(
    const float & _arg)
  {
    this->vel = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    task_2_interfaces::msg::JointData_<ContainerAllocator> *;
  using ConstRawPtr =
    const task_2_interfaces::msg::JointData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      task_2_interfaces::msg::JointData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      task_2_interfaces::msg::JointData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__task_2_interfaces__msg__JointData
    std::shared_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__task_2_interfaces__msg__JointData
    std::shared_ptr<task_2_interfaces::msg::JointData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JointData_ & other) const
  {
    if (this->center != other.center) {
      return false;
    }
    if (this->vel != other.vel) {
      return false;
    }
    return true;
  }
  bool operator!=(const JointData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JointData_

// alias to use template instance with default allocator
using JointData =
  task_2_interfaces::msg::JointData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace task_2_interfaces

#endif  // TASK_2_INTERFACES__MSG__DETAIL__JOINT_DATA__STRUCT_HPP_
