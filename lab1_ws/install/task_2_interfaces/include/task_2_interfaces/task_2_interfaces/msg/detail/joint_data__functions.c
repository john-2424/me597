// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from task_2_interfaces:msg/JointData.idl
// generated code does not contain a copyright notice
#include "task_2_interfaces/msg/detail/joint_data__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `center`
#include "geometry_msgs/msg/detail/point32__functions.h"

bool
task_2_interfaces__msg__JointData__init(task_2_interfaces__msg__JointData * msg)
{
  if (!msg) {
    return false;
  }
  // center
  if (!geometry_msgs__msg__Point32__init(&msg->center)) {
    task_2_interfaces__msg__JointData__fini(msg);
    return false;
  }
  // vel
  return true;
}

void
task_2_interfaces__msg__JointData__fini(task_2_interfaces__msg__JointData * msg)
{
  if (!msg) {
    return;
  }
  // center
  geometry_msgs__msg__Point32__fini(&msg->center);
  // vel
}

bool
task_2_interfaces__msg__JointData__are_equal(const task_2_interfaces__msg__JointData * lhs, const task_2_interfaces__msg__JointData * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // center
  if (!geometry_msgs__msg__Point32__are_equal(
      &(lhs->center), &(rhs->center)))
  {
    return false;
  }
  // vel
  if (lhs->vel != rhs->vel) {
    return false;
  }
  return true;
}

bool
task_2_interfaces__msg__JointData__copy(
  const task_2_interfaces__msg__JointData * input,
  task_2_interfaces__msg__JointData * output)
{
  if (!input || !output) {
    return false;
  }
  // center
  if (!geometry_msgs__msg__Point32__copy(
      &(input->center), &(output->center)))
  {
    return false;
  }
  // vel
  output->vel = input->vel;
  return true;
}

task_2_interfaces__msg__JointData *
task_2_interfaces__msg__JointData__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__msg__JointData * msg = (task_2_interfaces__msg__JointData *)allocator.allocate(sizeof(task_2_interfaces__msg__JointData), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(task_2_interfaces__msg__JointData));
  bool success = task_2_interfaces__msg__JointData__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
task_2_interfaces__msg__JointData__destroy(task_2_interfaces__msg__JointData * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    task_2_interfaces__msg__JointData__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
task_2_interfaces__msg__JointData__Sequence__init(task_2_interfaces__msg__JointData__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__msg__JointData * data = NULL;

  if (size) {
    data = (task_2_interfaces__msg__JointData *)allocator.zero_allocate(size, sizeof(task_2_interfaces__msg__JointData), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = task_2_interfaces__msg__JointData__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        task_2_interfaces__msg__JointData__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
task_2_interfaces__msg__JointData__Sequence__fini(task_2_interfaces__msg__JointData__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      task_2_interfaces__msg__JointData__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

task_2_interfaces__msg__JointData__Sequence *
task_2_interfaces__msg__JointData__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__msg__JointData__Sequence * array = (task_2_interfaces__msg__JointData__Sequence *)allocator.allocate(sizeof(task_2_interfaces__msg__JointData__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = task_2_interfaces__msg__JointData__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
task_2_interfaces__msg__JointData__Sequence__destroy(task_2_interfaces__msg__JointData__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    task_2_interfaces__msg__JointData__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
task_2_interfaces__msg__JointData__Sequence__are_equal(const task_2_interfaces__msg__JointData__Sequence * lhs, const task_2_interfaces__msg__JointData__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!task_2_interfaces__msg__JointData__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
task_2_interfaces__msg__JointData__Sequence__copy(
  const task_2_interfaces__msg__JointData__Sequence * input,
  task_2_interfaces__msg__JointData__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(task_2_interfaces__msg__JointData);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    task_2_interfaces__msg__JointData * data =
      (task_2_interfaces__msg__JointData *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!task_2_interfaces__msg__JointData__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          task_2_interfaces__msg__JointData__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!task_2_interfaces__msg__JointData__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
