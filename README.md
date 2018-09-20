# Space planning test model based on subdivision

The `room_data.json` file specifies the problem definition including room sizes and adjacency criteria. To create a space plan import the `get_layout` function from the `space_planning` library. The function expects 5 inputs (n represents the number of rooms specified in the problem definition):

- room_def: problem definition imported from `room_data.json` file
- split_list: list of (n-2) floats in domain [0.0-1.0] which determines the order of the subdivision tree
- dir_list: list of (n-1) boolean variables [0 or 1] which determines the direction of each split
- room_order: an ordering of (n) sequential integers starting at 0 which determines room assignment
- min_opening: the minimum size of connections between rooms and to the outside (for example the width of a door)

The function generates 3 outputs:

- the walls as lines represented by its endpoints [[x1, y1], [x2, y2]]
- the adjacency score as number of adjacency rules broken (0 represents a perfect design) - this should be minimized during optimization
- the aspect score as deviation from expected aspect ratios (0.0 represents perfect aspect match) - this should be minimized during optimization
