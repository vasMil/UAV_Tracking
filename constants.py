from typing import Optional, Tuple
import math

#########################################################################################
####################################### SIMULATION ######################################
#########################################################################################
# Vehicle names as defined in setting.json file
EGO_UAV_NAME = "EgoUAV"
LEADING_UAV_NAME = "LeadingUAV"

# The port number airsim if configured to listen to
PORT = 41451
# The seed used for the LeadingUAV random movements
LEADING_UAV_SEED: Optional[int] = 10

# Minimum acceptable error
EPS = 1e-5

# The clock speed as defined in settings.json
CLOCK_SPEED = 1

# Pawn size
PAWN_SIZE_X = 0.98
PAWN_SIZE_Y = 0.98
PAWN_SIZE_Z = 0.29

# Camera settings, change this if you change the defaults in setting.json (or vice versa)
IMG_HEIGHT = 144
IMG_WIDTH = 256
ASPECT_RATIO = IMG_WIDTH / IMG_HEIGHT  # (=16:9)

# Found by using simGetCameraInfo()
HORIZONTAL_FOV_DEG = 89.90362548828125
HORIZ_FOV = math.radians(HORIZONTAL_FOV_DEG)
# How to calculate the vertical FOV:
# https://github.com/microsoft/AirSim/issues/902
VERT_FOV = math.radians((IMG_HEIGHT / IMG_WIDTH) * HORIZONTAL_FOV_DEG)
CAMERA_OFFSET_X = 0.4599999785423279
# The focal length of our camera has already been calculated as
# F = (P * D) / W, where
# P is the width of the object in pixels,
# D is the actual distance of the camera, from that object,
# W is the width of the object in meters
# We placed the two UAVs the one infront of the other, with 3.5
# meters between their centers. Thus the actual distance between the
# camera and the back of the leadingUAV is 3.5 - camera_offset_x - pawn_size_x/2.
FOCAL_LENGTH_X = 46 * (3.5 - CAMERA_OFFSET_X - PAWN_SIZE_X/2) / PAWN_SIZE_Y
FOCAL_LENGTH_Y = 13 * (3.5 - CAMERA_OFFSET_X - PAWN_SIZE_X/2) / PAWN_SIZE_Z

EGO_CAMERA_NAME = "0"

IMG_RESOLUTION_INCR_FACTOR = 3
MOVEMENT_PLOT_MIN_RANGE = 10
SCORE_THRESHOLD = 0.1

#########################################################################################
##################################### DATA GENERATION ###################################
#########################################################################################
# Bounds for random distance between the two UAVs
MIN_DIST = 1.5
MAX_DIST = 10
# Box to allow random movement of the egoUAV in
RAND_MOVE_BOX_X: Tuple[float, float] = (-10., 10.,)
RAND_MOVE_BOX_Y: Tuple[float, float] = (-10., 10.,)
RAND_MOVE_BOX_Z: Tuple[float, float] = (-10., -1.,) # Min dist from the ground is -1 (i.e. 1m above ground level)

# The position calculated inside create_sample is slightly different
# by the one calculated inside generate_training_data,
# because when capturing those the UAVs are not completely stationary.
# If you want to minimize this allowed threshold you will have to
# increase wait_stationarity.
MEASUREMENT_THREASHOLD = (0.1, 0.1, 0.1)
FILENAME_LEADING_ZEROS = 6

#########################################################################################
########################################## GRAPHS #######################################
#########################################################################################
STATUS_COLORS = ["darkred", "indianred", "orangered", "salmon", "green", "mediumorchid", "blueviolet"]
