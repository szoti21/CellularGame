
#game settings
GRID_SIZE = 10
NUMBER_OF_OUTPUTS = 4 #num of moves
NUMBER_OF_INPUTS = GRID_SIZE * GRID_SIZE + 2 # grid dimension + days + energy

INITIAL_ENERGY = 10
DAY_COUNT = 0

TREE_DENSITY = 0.7
LION_DENSITY = 0.1

# 0 is always the empty value
PLAYER_VALUE = 1
TREE_VALUE = 2
LION_VALUE = 3

#heuristics
MEETING_WITH_A_LION_VALUE = -1000
EATING_TREE_VALUE = 200

ENERGY_FROM_TREE = 2
DAY_RATIO = 10

# training settings
NUM_OF_TRAININGS = 500000

# model saving
MODEL_PATH = "../resources/models/training_full_game.pth"
LOAD_MODEL = True # true: load saved model and refine it, false: to start the training over
SAVE_MODEL_AFTER_EVERY_X_TURNS = 300