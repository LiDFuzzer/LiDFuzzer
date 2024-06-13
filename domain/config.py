"""

CONFIGURATION FILE FOR LiDAR TEST GENERATOR


"""

# ----------------------------------------------------
# Configurable mutation parameters

# ---

# ADD
ADD_PLACEMENT_ATTEMPT_LIMIT = 10
ADD_PLACEMENT_LINE_POINTS = 40
ADD_ALLOWED_IN_FRUTSUM = 10
ADD_OVER_GROUND = 3

# ---

# REMOVE
REMOVE_POINTS_ABOVE_LIMIT = 30


# ---

# DEFORM
DEFORM_MIN_CHANGE = 0.05
DEFORM_MAX_CHANGE = 0.12
DEFORM_MU = 0.05
DEFORM_SIGMA = 0.04

# ---

# INTENSITY
INTENSITY_IGNORE_THRESHOLD = 0.8
INTENSITY_MIN_CHANGE = 0.1
INTENSITY_MAX_CHANGE = 0.3

# ---

# SCALE
SCALE_MESH_POINTS_THRESHOLD = 10000
SCALE_MESH_RADII = 0.15
SCALE_AMOUNT = 1.05
SCALE_MIN_POINTS = 20

# ---

# SIGN REPLACE
SIGN_MIN_POLE_POINTS = 5
SIGN_MIN_SIGN_POINTS = 5
SIGN_MIN_ALLOWED_SIGN_HEIGHT = -1
SIGN_SIZE_CHECK = 2
SIGN_MIN_POINTS = 15


# ----------------------------------------------------


Carrule = {
    "06": {
        # 80: [72, 48, 40, 44, 49],
        18: [40, 48, 72],
        15: [48, 40, 72, 44],
        20: [40, 48, 49, 72, 44],
        10: [40, 48, 44, 72, 49],
        81: [72, 48, 40, 44, 49],
        11: [48, 72, 40, 44],
        30: [40, 48, 49, 72, 44],
        31: [40, 48, 49, 72, 44]
    },
    "04": {
        20: [40, 44, 49, 72], 
        10: [40, 49, 44, 72, 48], 
        # 80: [72, 48, 44, 49, 40], 
        81: [40, 48, 49, 72, 44]
    },
    "03": {
        # 80: [48, 72, 44, 40, 49], 
        10: [44, 40, 48, 72],  
        30: [40, 44, 48], 
        81: [48, 44, 72, 40], 
        20: [44, 48, 72, 40], 
        11: [72, 48, 40], 
        18: [40]
        }
}




