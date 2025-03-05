# solve.py
import torch
import numpy as np
import time
from game import GameEngine
from rlmodel import DQN
from config import NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, MODEL_PATH


def solve_game():
    # creates the game engine
    env = GameEngine()
    state_dim = NUMBER_OF_INPUTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = DQN(state_dim, NUMBER_OF_OUTPUTS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode (no training)

    state = env.reset()
    done = False
    step_count = 0

    # Track the agent's path
    path = [env.player_pos.copy()]

    print("\n🔹 Starting Grid:")
    env.print_grid()

    while not done:
        action = None
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)


        # Get the best action from Q-values
        available_actions = env.all_valid_actions()
        with torch.no_grad():
            moves = torch.argsort(model(state_tensor), stable=True, descending=True)[0]
            for move in moves:
                if move in available_actions:
                    action = move
                    break
            if action is None:
                action = torch.argmax(model(state_tensor)).item()

        # Take action in environment
        state, _, done = env.step(action)

        # Track path
        path.append(env.player_pos.copy())
        step_count += 1

        print(f"\n🔹 Step {step_count}: Agent moved {['Up', 'Down', 'Left', 'Right'][action]}")
        env.print_grid()
        time.sleep(0.5)  # Pause for visibility

    print("\n✅ Agent reached the exit in", step_count, "steps!")
    print("🔹 Path Taken:", path)


# Run the solver
solve_game()
