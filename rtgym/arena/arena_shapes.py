import torch


def generate_rectangle_arena(sr, **kwargs):
    dimensions = kwargs.get("dimensions", [100, 100])
    assert len(dimensions) == 2, "dimensions must be a list of 2 elements"

    # convert dimensions from cm to pixel
    dimensions = [int(dimensions[i] / sr) for i in range(2)]  # pixel

    border = 5
    arena_map = torch.zeros(dimensions)
    arena_map = torch.nn.functional.pad(
        arena_map, (border, border, border, border), mode="constant", value=1
    )

    return arena_map


def generate_carpenter_rooms_arena(sr, **kwargs):
    room_width = int(kwargs.get("room_width", 90) / sr)
    room_height = int(kwargs.get("room_height", 90) / sr)
    corridor_width = int(kwargs.get("corridor_width", 40) / sr)
    wall_thickness = int(kwargs.get("wall_thickness", 1) / sr)
    opening_width = int(kwargs.get("opening_width", 20) / sr)
    border = 5

    assert room_width > 0 and room_height > 0, "rooms must have positive dims"
    assert corridor_width > 0, "corridor_width must be positive"
    assert opening_width > 0, "opening_width must be positive"
    assert opening_width < room_width, (
        f"opening_width ({opening_width}) must be < room_width ({room_width})"
    )

    total_width = 2 * room_width + wall_thickness
    total_height = corridor_width + wall_thickness + room_height
    arena_map = torch.zeros((total_height, total_width), dtype=torch.float32)

    # Row ranges
    wall_r0 = corridor_width
    wall_r1 = corridor_width + wall_thickness
    room_r0 = wall_r1
    room_r1 = room_r0 + room_height

    # North wall between the corridor and the two rooms, full width.
    arena_map[wall_r0:wall_r1, :] = 1

    # Solid dividing wall between rooms A and B, full room height.
    div_c0 = room_width
    div_c1 = div_c0 + wall_thickness
    arena_map[room_r0:room_r1, div_c0:div_c1] = 1

    # Opening in each room's north wall, centered on the sub-room. 
    half = opening_width // 2
    b_open_center = room_width // 2  # center of room B
    a_open_center = div_c1 + room_width // 2  # center of room A

    b_c0 = max(0, b_open_center - half)
    b_c1 = min(div_c0, b_open_center + (opening_width - half))
    a_c0 = max(div_c1, a_open_center - half)
    a_c1 = min(total_width, a_open_center + (opening_width - half))

    arena_map[wall_r0:wall_r1, b_c0:b_c1] = 0
    arena_map[wall_r0:wall_r1, a_c0:a_c1] = 0

    # Enclose with a wall border.
    arena_map = torch.nn.functional.pad(
        arena_map, (border, border, border, border), mode="constant", value=1
    )

    if kwargs.get("vertical"):
        arena_map = torch.rot90(arena_map, 1, [0, 1])

    return arena_map


def generate_box_arena(sr, **kwargs):
    dimensions = kwargs.get("dimensions", [100, 100, 100])
    assert len(dimensions) == 3, "dimensions must be a list of 3 elements"

    # Convert dimensions from spatial units to pixels (free interior).
    dimensions = [int(dimensions[i] / sr) for i in range(3)]

    border = 5
    arena_map = torch.zeros(dimensions)
    # F.pad tuple order for 3D: (W_left, W_right, H_left, H_right, D_left, D_right)
    arena_map = torch.nn.functional.pad(
        arena_map,
        (border, border, border, border, border, border),
        mode="constant",
        value=1,
    )
    return arena_map


def generate_hairpin_arena(sr, **kwargs):
    n_alleys = int(kwargs.get("n_alleys", 10))
    alley_width = int(kwargs.get("alley_width", 15) / sr)
    alley_height = int(kwargs.get("alley_height", 100) / sr)
    wall_thickness = int(kwargs.get("wall_thickness", 1) / sr)
    turn_gap = int(kwargs.get("turn_gap", kwargs.get("alley_width", 15)) / sr)
    border = 5

    assert n_alleys >= 2, f"n_alleys must be >= 2, got {n_alleys}"
    assert turn_gap < alley_height, (
        f"turn_gap ({turn_gap}) must be smaller than alley_height ({alley_height})"
    )

    height = alley_height
    width = n_alleys * alley_width + (n_alleys - 1) * wall_thickness
    arena_map = torch.zeros((height, width), dtype=torch.float32)

    # Walls between alleys.
    for i in range(n_alleys - 1):
        col_start = (i + 1) * alley_width + i * wall_thickness
        col_end = col_start + wall_thickness
        if i % 2 == 0:
            arena_map[turn_gap:, col_start:col_end] = 1  # gap at top
        else:
            arena_map[: height - turn_gap, col_start:col_end] = 1  # gap at bottom

    # Border (walls on all four sides)
    arena_map = torch.nn.functional.pad(
        arena_map, (border, border, border, border), mode="constant", value=1
    )

    if kwargs.get("vertical"):
        arena_map = torch.rot90(arena_map, 1, [0, 1])

    return arena_map


def hairpin_waypoints(
    sr,
    n_alleys=10,
    alley_width=15,
    alley_height=100,
    wall_thickness=2,
    turn_gap=None,
    border=5,
    direction="forward",
):
    alley_width_px = int(alley_width / sr)
    alley_height_px = int(alley_height / sr)
    wall_thickness_px = int(wall_thickness / sr)
    if turn_gap is None:
        turn_gap_px = alley_width_px
    else:
        turn_gap_px = int(turn_gap / sr)

    col_centers = [
        border + i * alley_width_px + i * wall_thickness_px + alley_width_px // 2
        for i in range(n_alleys)
    ]
    # Rows inside the alleys — stay a few pixels inside the turn gap so the
    # waypoint is reachable without clipping the wall.
    margin = max(2, turn_gap_px // 3)
    top_row = border + margin
    bottom_row = border + alley_height_px - 1 - margin

    waypoints = [(bottom_row, col_centers[0])]
    for i in range(n_alleys):
        if i % 2 == 0:
            # Going up alley i
            waypoints.append((top_row, col_centers[i]))
            if i + 1 < n_alleys:
                waypoints.append((top_row, col_centers[i + 1]))
        else:
            # Going down alley i
            waypoints.append((bottom_row, col_centers[i]))
            if i + 1 < n_alleys:
                waypoints.append((bottom_row, col_centers[i + 1]))

    if direction == "backward":
        waypoints = list(reversed(waypoints))
    elif direction != "forward":
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction}")

    return waypoints
