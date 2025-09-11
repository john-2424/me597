// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from task_2_interfaces:srv/JointState.idl
// generated code does not contain a copyright notice
#include "task_2_interfaces/srv/detail/joint_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
task_2_interfaces__srv__JointState_Request__init(task_2_interfaces__srv__JointState_Request * msg)
{
  if (!msg) {
    return false;
  }
  // x
  // y
  // z
  return true;
}

void
task_2_interfaces__srv__JointState_Request__fini(task_2_interfaces__srv__JointState_Request * msg)
{
  if (!msg) {
    return;
  }
  // x
  // y
  // z
}

bool
task_2_interfaces__srv__JointState_Request__are_equal(const task_2_interfaces__srv__JointState_Request * lhs, const task_2_interfaces__srv__JointState_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
    return false;
  }
  return true;
}

bool
task_2_interfaces__srv__JointState_Request__copy(
  const task_2_interfaces__srv__JointState_Request * input,
  task_2_interfaces__srv__JointState_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  return true;
}

task_2_interfaces__srv__JointState_Request *
task_2_interfaces__srv__JointState_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__srv__JointState_Request * msg = (task_2_interfaces__srv__JointState_Request *)allocator.allocate(sizeof(task_2_interfaces__srv__JointState_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(task_2_interfaces__srv__JointState_Request));
  bool success = task_2_interfaces__srv__JointState_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
task_2_interfaces__srv__JointState_Request__destroy(task_2_interfaces__srv__JointState_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    task_2_interfaces__srv__JointState_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
task_2_interfaces__srv__JointState_Request__Sequence__init(task_2_interfaces__srv__JointState_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__srv__JointState_Request * data = NULL;

  if (size) {
    data = (task_2_interfaces__srv__JointState_Request *)allocator.zero_allocate(size, sizeof(task_2_interfaces__srv__JointState_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = task_2_interfaces__srv__JointState_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        task_2_interfaces__srv__JointState_Request__fini(&data[i - 1]);
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
task_2_interfaces__srv__JointState_Request__Sequence__fini(task_2_interfaces__srv__JointState_Request__Sequence * array)
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
      task_2_interfaces__srv__JointState_Request__fini(&array->data[i]);
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

task_2_interfaces__srv__JointState_Request__Sequence *
task_2_interfaces__srv__JointState_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__srv__JointState_Request__Sequence * array = (task_2_interfaces__srv__JointState_Request__Sequence *)allocator.allocate(sizeof(task_2_interfaces__srv__JointState_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = task_2_interfaces__srv__JointState_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
task_2_interfaces__srv__JointState_Request__Sequence__destroy(task_2_interfaces__srv__JointState_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    task_2_interfaces__srv__JointState_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
task_2_interfaces__srv__JointState_Request__Sequence__are_equal(const task_2_interfaces__srv__JointState_Request__Sequence * lhs, const task_2_interfaces__srv__JointState_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!task_2_interfaces__srv__JointState_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
task_2_interfaces__srv__JointState_Request__Sequence__copy(
  const task_2_interfaces__srv__JointState_Request__Sequence * input,
  task_2_interfaces__srv__JointState_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(task_2_interfaces__srv__JointState_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    task_2_interfaces__srv__JointState_Request * data =
      (task_2_interfaces__srv__JointState_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!task_2_interfaces__srv__JointState_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          task_2_interfaces__srv__JointState_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!task_2_interfaces__srv__JointState_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
task_2_interfaces__srv__JointState_Response__init(task_2_interfaces__srv__JointState_Response * msg)
{
  if (!msg) {
    return false;
  }
  // valid
  return true;
}

void
task_2_interfaces__srv__JointState_Response__fini(task_2_interfaces__srv__JointState_Response * msg)
{
  if (!msg) {
    return;
  }
  // valid
}

bool
task_2_interfaces__srv__JointState_Response__are_equal(const task_2_interfaces__srv__JointState_Response * lhs, const task_2_interfaces__srv__JointState_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // valid
  if (lhs->valid != rhs->valid) {
    return false;
  }
  return true;
}

bool
task_2_interfaces__srv__JointState_Response__copy(
  const task_2_interfaces__srv__JointState_Response * input,
  task_2_interfaces__srv__JointState_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // valid
  output->valid = input->valid;
  return true;
}

task_2_interfaces__srv__JointState_Response *
task_2_interfaces__srv__JointState_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__srv__JointState_Response * msg = (task_2_interfaces__srv__JointState_Response *)allocator.allocate(sizeof(task_2_interfaces__srv__JointState_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(task_2_interfaces__srv__JointState_Response));
  bool success = task_2_interfaces__srv__JointState_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
task_2_interfaces__srv__JointState_Response__destroy(task_2_interfaces__srv__JointState_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    task_2_interfaces__srv__JointState_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
task_2_interfaces__srv__JointState_Response__Sequence__init(task_2_interfaces__srv__JointState_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__srv__JointState_Response * data = NULL;

  if (size) {
    data = (task_2_interfaces__srv__JointState_Response *)allocator.zero_allocate(size, sizeof(task_2_interfaces__srv__JointState_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = task_2_interfaces__srv__JointState_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        task_2_interfaces__srv__JointState_Response__fini(&data[i - 1]);
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
task_2_interfaces__srv__JointState_Response__Sequence__fini(task_2_interfaces__srv__JointState_Response__Sequence * array)
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
      task_2_interfaces__srv__JointState_Response__fini(&array->data[i]);
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

task_2_interfaces__srv__JointState_Response__Sequence *
task_2_interfaces__srv__JointState_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  task_2_interfaces__srv__JointState_Response__Sequence * array = (task_2_interfaces__srv__JointState_Response__Sequence *)allocator.allocate(sizeof(task_2_interfaces__srv__JointState_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = task_2_interfaces__srv__JointState_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
task_2_interfaces__srv__JointState_Response__Sequence__destroy(task_2_interfaces__srv__JointState_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    task_2_interfaces__srv__JointState_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
task_2_interfaces__srv__JointState_Response__Sequence__are_equal(const task_2_interfaces__srv__JointState_Response__Sequence * lhs, const task_2_interfaces__srv__JointState_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!task_2_interfaces__srv__JointState_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
task_2_interfaces__srv__JointState_Response__Sequence__copy(
  const task_2_interfaces__srv__JointState_Response__Sequence * input,
  task_2_interfaces__srv__JointState_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(task_2_interfaces__srv__JointState_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    task_2_interfaces__srv__JointState_Response * data =
      (task_2_interfaces__srv__JointState_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!task_2_interfaces__srv__JointState_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          task_2_interfaces__srv__JointState_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!task_2_interfaces__srv__JointState_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
