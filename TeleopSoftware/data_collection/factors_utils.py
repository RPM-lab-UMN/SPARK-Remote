import numpy as np

# Dictionary defining the reachability line for each table height 
# (how far the robot arm can reach)
# Format: height: {'m': slope, 'c': y value at y-intercept (x=0), 'x_thresh': x value at x-intercept (y=0)}
# Invalid condition: (y <= mx + c) and (x >= x_thresh)
# CORRECT
REACHABILITY_BOUNDARIES = {
    1: {"m": 0.85, "c": -0.17, "x_thresh": 0.2},
    2: {"m": 0.867, "c": -0.2167, "x_thresh": 0.15},
    3: {"m": 0.857, "c": -0.257, "x_thresh": 0.0},
}
# WRONG
# REACHABILITY_BOUNDARIES = {
#     1: {'m': 0.857, 'c': -0.257, 'x_thresh': 0.0}, # y = 0.857x - 0.257
#     2: {'m': 0.867, 'c': -0.2167, 'x_thresh': 0.15}, # y = 0.867x - 0.2167 
#     3: {'m': 0.85, 'c': -0.17, 'x_thresh': 0.2}, # y = 0.85x - 0.17
# }

def is_valid_point(point, table_height):
    """
    Filters out points past UR5 arm's reachability based on table height.
    Returns True if point is reachable, False otherwise.
    """
    # Get coordinates
    # Handle both numpy scalars (which have .item()) and standard floats
    x = point[0].item() if hasattr(point[0], 'item') else point[0]
    y = point[1].item() if hasattr(point[1], 'item') else point[1]
    
    # Retrieve constraints for this specific table height
    # Default to height 4 params if height is not found (safety fallback)
    params = REACHABILITY_BOUNDARIES[table_height]
    m = params['m']
    c = params['c']
    x_thresh = params['x_thresh']

    # Check validity
    # A point is 'invalid' if it is both 
    # 1) below the diagonal line y = mx + c 
    # 2) AND to the right of x_thresh 
    is_invalid = (y <= m * x + c) and (x >= x_thresh)
    
    return not is_invalid


# def gen_factors():
    """
    FOR THE TASKS: 'pick up the blue block' and 'set the cup upright'
    Generates random factors for scene setup, ensuring reachability.
    """
    while True:
        # 1. Generate random factors candidates
        block_x = np.random.choice(np.linspace(0.5, 1.0, num=6)) # change for diff quadrant
        block_y = np.random.choice(np.linspace(0.5, 1.0, num=6)) # change for diff quadrant

        # Select table height (you can randomize this among choices or keep it fixed)
        # table_height = np.random.choice([1, 2, 3, 4]) 
        table_height = 3

        camera_viewpoint = "right"

        # 2. Check reachability
        if is_valid_point((block_x, block_y), table_height):
            # If valid, construct dictionary and break the loop
            factors = {
                "block_x": block_x, 
                "block_y": block_y, 
                'table_height': table_height,
                "camera_viewpoint": camera_viewpoint,
            }
            
            print("====== FACTOR VALUES FOR THE NEXT DEMO ======")
            print(f" Object Position (Grid): x={factors['block_x']:.3f}, y={factors['block_y']:.3f}")
            # print(f" Table Height: {factors['table_height']}")
            return factors


def gen_factors():
    """
    FOR THE TASK: 'put the green block in the pot'
    Generates random factors for scene setup, ensuring reachability.
    """
    # Define a minimum distance to prevent overlap
    MIN_DIST_BLOCK_LIDPOT = 0.15
    # MIN_DIST_LID_POT = 0.2
    pot_x = 0.5
    pot_y = 0.5
    while True:
        # Generate factors
        # block_x = np.random.choice(np.linspace(0.0, 0.5, num=6)) # change for diff quadrant
        # block_y = np.random.choice(np.linspace(0.0, 0.5, num=6)) # change for diff quadrant
        block_x = 0.4
        block_y = 0.7

        # lid_on = np.random.choice([True, False])
        # if lid_on:
        #     # If the lid is on the pot, we assume a fixed pot position (0.5, 0.5)
        #     lid_x, lid_y = pot_x, pot_y
        # else:
        #     lid_x = np.random.choice(np.linspace(0.0, 1.0, num=6))
        #     lid_y = np.random.choice(np.linspace(0.0, 1.0, num=6))
        
        # Select table height (you can randomize this among choices or keep it fixed)
        # table_height = np.random.choice([1, 2, 3, 4]) 
        table_height = 3

        camera_viewpoint = "back"

        # Calculate Euclidean distances
        # distance_block_lid = np.sqrt((block_x - lid_x)**2 + (block_y - lid_y)**2)
        distance_block_pot = np.sqrt((block_x - pot_x)**2 + (block_y - pot_y)**2)

        # Check reachability
        reach_ok = is_valid_point((block_x, block_y), 3)
        # reach_ok = is_valid_point((block_x, block_y), 3) and is_valid_point((lid_x, lid_y), 3)

        # Block should always be away from Lid and Pot
        # overlap_ok_block_lid = distance_block_lid > MIN_DIST_BLOCK_LIDPOT
        overlap_ok_block_pot = distance_block_pot > MIN_DIST_BLOCK_LIDPOT # reusing block/pot threshold

        # Only check Lid-to-Pot distance if the lid is NOT on
        # if not lid_on:
        #     distance_lid_pot = np.sqrt((lid_x - pot_x)**2 + (lid_y - pot_y)**2)
        #     overlap_ok_lid_pot = distance_lid_pot > MIN_DIST_LID_POT
        # else:
        #     # If lid is on, we don't care about the lid-pot distance check
        #     overlap_ok_lid_pot = True 

        if reach_ok and overlap_ok_block_pot:
        # if reach_ok and overlap_ok_block_lid and overlap_ok_block_pot and overlap_ok_lid_pot:
            # If valid, construct dictionary and break the loop
            factors = {
                "block_x": block_x, 
                "block_y": block_y,
                # "lid_x": lid_x,
                # "lid_y": lid_y,
                # "lid_on": lid_on,
                'table_height': table_height,
                "camera_viewpoint": camera_viewpoint,
            }
            
            print("====== FACTOR VALUES FOR THE NEXT DEMO ======")
            print(f" Block Position (Grid): x={factors['block_x']:.3f}, y={factors['block_y']:.3f}")
            # if not lid_on:
            #     print(f" Lid Position (Grid): x={factors['lid_x']:.3f}, y={factors['lid_y']:.3f}")
            # else:
            #     print("Lid is on")
            # print(f" Table Height: {factors['table_height']}")
            return factors