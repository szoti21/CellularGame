
#game settings
GRID_SIZE = 5
NUMBER_OF_OUTPUTS = 4 #num of moves
NUMBER_OF_INPUTS = GRID_SIZE * GRID_SIZE + 2 # grid dimension + days + energy

INITIAL_ENERGY = 10
DAY_COUNT = 0

TREE_DENSITY = 0.4
LION_DENSITY = 0.2

# 0 is always the empty value
PLAYER_VALUE = 1
TREE_VALUE = 2
LION_VALUE = 3

#heuristics
INVALID_MOVE_VALUE = -1000
MEETING_WITH_A_LION_VALUE = -100

ENERGY_FROM_TREE = 2
DAY_RATIO = 1
ENERGY_RATIO = 2

# training settings
NUM_OF_TRAININGS = 50000

# model saving
MODEL_PATH = "../resources/models/training_with_trees.pth"
LOAD_MODEL = True # true: load saved model and refine it, false: to start the training over
SAVE_MODEL_AFTER_EVERY_X_TURNS = 50