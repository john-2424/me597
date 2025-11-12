from enum import Enum

class SearchStates(Enum):
    none = None
    rotate_z_d = "rotate_z_d"
    move_x_d = "move_x_d"
    find_gaps = "find_gaps"
    pick_a_gap = "pick_a_gap"
    traverse_the_gap = "traverse_the_gap"