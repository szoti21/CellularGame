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
        self.player_pos = [GRID_SIZE // 2, GRID_SIZE // 2]

        # random trees - excluding the player
        num_trees = int(TREE_DENSITY * GRID_SIZE * GRID_SIZE)
        #num_trees = 2
        self.trees = self.random_positions(num_trees, exclude=tuple(self.player_pos))
        #self.trees = self.random_positions(num_trees, exclude=tuple(self.player_pos), fixed_positions=[(4, 5), (4, 1)])

        # random lions - excluding the player and trees
        num_lions = int(LION_DENSITY * GRID_SIZE * GRID_SIZE)
        #num_lions = 5
        self.lions = self.random_positions(num_lions, exclude=self.trees | {tuple(self.player_pos)})
        #self.lions = self.random_positions(num_lions, exclude=self.trees | {tuple(self.player_pos)}, fixed_positions=[(5, 4), (6, 5), (5, 6), (4, 4), (3, 5)])

        self.visible = -1 * np.ones((GRID_SIZE, GRID_SIZE))
        self.check_visibility()

        return self.get_state()

    def get_state(self):
        state = self.visible
        state[self.player_pos[0], self.player_pos[1]] = PLAYER_VALUE
        return np.append(state.flatten(), [self.energy, self.day])

    def check_visibility(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in directions:
            if self.player_pos[0] + x < 0:
                x = GRID_SIZE - 1
            elif self.player_pos[0] + x > GRID_SIZE - 1:
                x = -1 * (GRID_SIZE - 1)
            if self.player_pos[1] + y < 0:
                y = GRID_SIZE - 1
            elif self.player_pos[1] + y > GRID_SIZE - 1:
                y = -1 * (GRID_SIZE - 1)

            if (self.player_pos[0] + x, self.player_pos[1] + y) in self.trees:
                self.visible[self.player_pos[0] + x][self.player_pos[1] + y] = TREE_VALUE
            elif (self.player_pos[0] + x, self.player_pos[1] + y) in self.lions:
                self.visible[self.player_pos[0] + x][self.player_pos[1] + y] = LION_VALUE
            else :
                self.visible[self.player_pos[0] + x][self.player_pos[1] + y] = 0

    def refog(self):
        self.visible[self.visible == 0] = -1

    def step(self, action):
        x, y = self.player_pos

        if action == UP:
            x = x - 1
            if x < 0:
                x = GRID_SIZE-1
        elif action == DOWN:
            x = x + 1
            if x >= GRID_SIZE:
                x = 0
        elif action == LEFT:
            y = y - 1
            if y < 0:
                y = GRID_SIZE-1
        elif action == RIGHT:
            y = y + 1
            if y >= GRID_SIZE:
                y = 0

        reward = 0
        done = False

        self.day = self.day + 1
        self.energy = self.energy - 1

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
            self.refog()

        reward, done = self.get_score(is_lion, is_tree)
        self.player_pos = [x, y]
        self.check_visibility()
        return self.get_state(), reward, done

    def get_score(self, is_lion, is_tree):
        reward = 0
        done = False

        if self.energy <= 0:
            reward += self.day * DAY_RATIO
            if self.day == 10:
                reward -= 100
            done = True

        if is_lion:
            reward += MEETING_WITH_A_LION_VALUE
            reward += self.day * DAY_RATIO
            done = True

        if is_tree:
            reward += EATING_TREE_VALUE
            done = False

        #reward = reward + DAY_RATIO * self.day + ENERGY_RATIO * self.energy

        return reward, done

    def random_positions(self, count, exclude, fixed_positions=None):
        if fixed_positions:
            return set(fixed_positions)
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