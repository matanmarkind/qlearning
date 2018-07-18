from matplotlib import pyplot as plt
from enum import Enum

import numpy as np

import time

class Gridworld():
    class Color(Enum):
        """
        Exact values matter since RGB line up with the depth they refer to in
        and RGB image.
        """
        RED = 0  # -1 point
        GREEN = 1  # +1 point
        BLUE = 2  # self
        BLACK = 3  # 0 points

    class Action(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    def __init__(self, rows=5, cols=5, greens=3, reds=2):
        self.shape = (rows, cols)
        self.greens = greens
        self.reds = reds
        assert (greens + reds) < (rows * cols), \
            "Grid not large enough: shape={}, greens={}, reds={}.".format(
                self.shape, greens, reds)

        # Needed for live image showing.
        plt.ion()
        plt.show()

    def render(self):
        plt.imshow(self._grid_to_img(self.grid))
        plt.draw()
        plt.pause(np.finfo(np.double).tiny)

    def reset(self):
        self.grid = self._create_grid(self.shape)
        return self._grid_to_img(self.grid)

    def step(self, action):
        """
        Progress the game by taking an actiona nd moving the blue tile.
        :param action: int to be converted to self.Action
        :return: img, reward, done, info
        """
        action = self.Action(action)
        color = self.Color.BLACK  # color that blue moves onto.
        row, col = np.argwhere(self.grid == self.Color.BLUE)[0]
        if (action == self.Action.UP):
            color = self._move_to(row-1, col)
        elif (action == self.Action.DOWN):
            color = self._move_to(row+1, col)
        elif (action == self.Action.RIGHT):
            color = self._move_to(row, col+1)
        elif (action == self.Action.LEFT):
            color = self._move_to(row, col-1)

        reward = self._calc_reward(color)
        # Game doesn't have an end condition, so randomly decide if done.
        # 90% done by 100 turns.
        done = np.random.random() > (1 - .9) ** (1 / 100)  # .977
        # No extra info to give, but want to have a gym-like interface.
        return self._grid_to_img(self.grid), reward, done, {}

    @property
    def n_actions(self):
        """
        :return: Number of possible actions
        """
        return 4

    def sample_action(self):
        return np.random.randint(0, self.n_actions)

    def _create_grid(self, shape):
        grid = np.full(shape, self.Color.BLACK, dtype=self.Color)
        grid = self._place_color(grid, self.greens, self.Color.GREEN)
        grid = self._place_color(grid, self.reds, self.Color.RED)
        return self._place_color(grid, 1, self.Color.BLUE)

    def _place_color(self, grid, num, color):
        """
        Randomly place some amount of colors on top of unoccupied (black) elements.
        :param grid:
        :param num:
        :param color:
        :return:
        """
        assert True in np.isin(grid, self.Color.BLACK), \
            "No open tiles, infinite loop"
        while num > 0:
            row = np.random.randint(0, 5)
            col = np.random.randint(0, 5)
            if grid[row, col] == self.Color.BLACK:
                grid[row, col] = color
                num -= 1

        return grid


    def _grid_to_img(self, grid):
        """
        Takes in a grid which has enum Color values and makes it into an img.
        Assumes the colors are RGB and the value alligns with the depth of
        the appropriate matrix to convert for an image.
        (Red=0, Green=1, Blue=2)
        :param grid:
        :return:
        """
        scale = 10
        rows, cols = grid.shape
        img = np.zeros((rows * scale, cols * scale, 3), dtype=np.uint8)
        for row in range(rows):
            for col in range(cols):
                color = grid[row, col]
                if color == self.Color.BLACK:
                    continue
                r = scale * row
                c = scale * col
                img[r:(r+scale), c:(c+scale), color.value] = 255
        return img


    def _move_to(self, next_row, next_col):
        """
        Move the blue tile to the new row and column.
        :param row:
        :param col:
        :return:
        """
        if next_row < 0 or next_row >= self.shape[0] or \
                next_col < 0 or next_col >= self.shape[1]:
            return self.Color.BLACK
        at_row, at_col = np.argwhere(self.grid == self.Color.BLUE)[0]
        color = self.grid[next_row, next_col]
        self.grid[at_row, at_col] = self.Color.BLACK
        self.grid[next_row, next_col] = self.Color.BLUE
        self.grid = self._place_color(self.grid, 1, color)
        return color



    def _calc_reward(self, color):
        """
        Calculate reward for moving to a new color.
        :param color:
        :return:
        """
        if color == self.Color.RED:
            return -1
        elif color == self.Color.GREEN:
            return 1
        elif color == self.Color.BLACK:
            return 0
        elif color == self.Color.BLUE:
            assert False, "Internal Failure, can't move to BLUE"

    def __del__(self):
        # Turn off live plotting.
        plt.close()
        plt.ioff()


def main():
    # Play a random game.
    rows, cols = 10, 5
    gridworld = Gridworld(rows, cols)
    state = gridworld.reset()
    done = False
    turn = 0
    reward = 0
    while not done:
        gridworld.render()
        time.sleep(.2)
        state, r, done, _ = gridworld.step(gridworld.sample_action())
        if r != 0:
            print(r)
        turn += 1
        reward += r
    print('reward =', reward, 'turns =', turn)

if __name__ == '__main__':
    main()

