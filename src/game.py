import random
import numpy as np
from config import *
from moves import *


class GameEngine:
    def __init__(self):
        assert 0 <= TREE_DENSITY + LION_DENSITY < 0.9, "Total density should be less than 0.9!"
        self.reset()

    def reset(self):
        self.energy = INITIAL_ENERGY
        self.day = DAY_COUNT

        #rnadom position for the user
        self.player_pos = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]

        # random trees - excluding the player
        num_trees = int(TREE_DENSITY * GRID_SIZE * GRID_SIZE)
        self.trees = self.random_positions(num_trees, exclude=tuple(self.player_pos))

        # random lions - excluding the player and trees
        num_lions = int(LION_DENSITY * GRID_SIZE * GRID_SIZE)
        self.lions = self.random_positions(num_lions, exclude=self.trees | {tuple(self.player_pos)})

        return self.get_state()

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE))

        state[self.player_pos[0], self.player_pos[1]] = PLAYER_VALUE
        #print(f"DEBUG: self.trees = {self.trees}, Type: {type(self.trees)}")
        for x, y in self.trees:
            state[x, y] = TREE_VALUE

        for x, y in self.lions:
            state[x, y] = LION_VALUE

        return np.append(state.flatten(), [self.energy, self.day])

    def all_valid_actions(self):
        actions = []
        for i in [UP, DOWN, LEFT, RIGHT]:
            if self.valid_action(i):
                actions.append(i)
        return actions

    def valid_action(self, action):
        x, y = self.player_pos
        if action == UP and x == 0:
            return False
        elif action == DOWN and x == GRID_SIZE-1:
            return False
        elif action == LEFT and y == 0:
            return False
        elif action == RIGHT and y == GRID_SIZE-1:
            return False
        else:
            return True

    def step(self, action):
        x, y = self.player_pos

        if action == UP:
            x = x - 1
        elif action == DOWN:
            x = x + 1
        elif action == LEFT:
            y = y - 1
        elif action == RIGHT:
            y = y + 1

        reward = 0
        done = False

        self.day = self.day + 1
        self.energy = self.energy - 1

        # invalid move
        if (x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE):
            reward = INVALID_MOVE_VALUE
            done = True
        else:
            is_lion = False
            is_tree = False

            if (tuple([x, y]) in self.lions):
                is_lion = True

            if (tuple([x, y]) in self.trees):
                is_tree = True
                self.trees.remove(tuple([x, y]) )
                # hacky, next(iter gets the first touple element
                new_tree_pos = next(iter(self.random_positions(1, exclude=self.trees | self.lions | {tuple([x, y])})))
                self.trees.add(new_tree_pos)
                self.energy += ENERGY_FROM_TREE

            reward, done = self.get_score(is_lion, is_tree)
            self.player_pos = [x, y]

        return self.get_state(), reward, done

    def get_score(self, is_lion, is_tree):
        reward = 0
        done = False

        if self.energy <= 0:
            done = True

        if is_lion:
            reward += MEETING_WITH_A_LION_VALUE
            done = True

        if is_tree:
            done = False

        reward = reward + DAY_RATIO * self.day + ENERGY_RATIO * self.energy

        return reward, done

    def random_positions(self, count, exclude):
        positions = set()
        while len(positions) < count:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in exclude:
                positions.add(pos)
        return positions

    def print_grid(self):
        grid = np.full((GRID_SIZE, GRID_SIZE), "-", dtype=str)
        x_p, y_p = self.player_pos

        grid[x_p, y_p] = "A"

        for x, y in self.trees:
            grid[x, y] = "T"
        for x, y in self.lions:
            grid[x, y] = "X"

        # Print grid row by row
        print("\n🔹 Current Grid:")
        for row in grid:
            print(" ".join(row))

        # Print additional game info
        print(f"🌟 Energy: {self.energy} | 📅 Days Survived: {self.day}\n")