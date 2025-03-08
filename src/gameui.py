import pygame
import torch
import time
from game import GameEngine
from rlmodel import DQN
from config import *

CELL_SIZE = 100  # Size of each cell in pixels
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COLOR = (0, 0, 255)  # Blue
TREE_COLOR = (0, 255, 0)  # Green
LION_COLOR = (255, 165, 0)  # Orange
LINE_COLOR = (0, 0, 0)  # Black
FOG_COLOR = (50, 50, 50, 150)  # Gray with some transparency

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Grid Game - Test Play")
clock = pygame.time.Clock()

def draw_grid(env):
    screen.fill(WHITE)

    # Draw grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (x, 0), (x, HEIGHT), 2)
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, y), (WIDTH, y), 2)

    # Draw trees
    for x, y in env.trees:
        pygame.draw.rect(screen, TREE_COLOR,
                         (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw lions
    for x, y in env.lions:
        pygame.draw.rect(screen, LION_COLOR,
                         (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw agent
    agent_x, agent_y = env.player_pos
    pygame.draw.rect(screen, AGENT_COLOR,
                     (agent_y * CELL_SIZE, agent_x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw fog of war over unseen cells
    fog_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    fog_surface.fill(FOG_COLOR)

    for row in range(len(env.visible)):
        for col in range(len(env.visible[row])):
            if env.visible[row][col] == -1:
                screen.blit(fog_surface, (col * CELL_SIZE, row * CELL_SIZE))

    pygame.display.flip()
def test_play():
    env = GameEngine()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = DQN(NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode (no training)

    state = env.reset()
    done = False

    print("\n🔹 Starting AI-controlled game...")

    while not done:
        # Draw the grid before each move
        draw_grid(env)
        time.sleep(0.5)  # Pause for visibility

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)

        # Get the best action from Q-values
        # Get the best action from Q-values
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()

        # Take action in environment
        state, score, done = env.step(action)

    print(f"✅ AI reached the exit or game over! score: {score}")
    draw_grid(env)  # Final state
    time.sleep(2)

    # Quit the game
    pygame.quit()

test_play()