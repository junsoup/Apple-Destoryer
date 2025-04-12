import numpy as np
import math
import copy
import random

class Board:
    def __init__(self, width=17, height=10, seed=None, grid = None):
        if seed is None:
            seed = random.randint(1, 999999999)
        
        print('seed', seed)
        random.seed(seed)
        np.random.seed(seed)  # Set seed

        if grid is None:
            self.apple_grid = np.random.randint(1, 10, size=(height, width), dtype=np.int8)
        else:
            self.apple_grid = grid
        self.score = 0
        self.move_count = 0
        self.height, self.width = height, width
        self.valid_moves = None

    def print_board(self):
        for row in self.apple_grid:
            for cell in row:
                if cell == 0:
                    print(' ', end=' ')
                else:
                    print(cell, end=' ')
            print('')

    def move(self, pos1, pos2):
        self.valid_moves = None
        self.move_count += 1

        x1, y1 = pos1
        x2, y2 = pos2
        
        # Extract the selected region as a view (not a copy)
        selection = self.apple_grid[x1:x2+1, y1:y2+1]
        
        # Calculate sum of selection
        selection_sum = np.sum(selection)
        
        if selection_sum == 10:
            # Calculate points by counting non-zero cells
            points_earned = np.count_nonzero(selection)
            
            # Set the entire selection to zero in one operation
            selection.fill(0)
            
            self.score += points_earned
            return points_earned
        
        return 0

    def isClear(self):
        # Faster way to check if all elements are zero
        return not np.any(self.apple_grid)
    
    def get_valid_moves(self):
        """Generate all possible valid moves using cumulative sum for faster lookup"""
        if self.valid_moves:
            return self.valid_moves
        valid_moves = []
        prefix_sum = self.apple_grid.cumsum(axis=0).cumsum(axis=1)  # Precompute sums
        
        for x1 in range(self.height):
            for y1 in range(self.width):
                if self.apple_grid[x1, y1] == 0:
                    continue
                
                for x2 in range(x1, self.height):
                    for y2 in range(y1, self.width):
                        if self.apple_grid[x2, y2] == 0:
                            continue

                        # Compute the sum using prefix sum lookup
                        total = prefix_sum[x2, y2]
                        if x1 > 0:
                            total -= prefix_sum[x1 - 1, y2]
                        if y1 > 0:
                            total -= prefix_sum[x2, y1 - 1]
                        if x1 > 0 and y1 > 0:
                            total += prefix_sum[x1 - 1, y1 - 1]

                        if total == 10:
                            valid_moves.append(((x1, y1), (x2, y2)))
                            break  # No need to expand further in this direction
                        elif total > 10:
                            break  # Stop early
        self.valid_moves = valid_moves  
        return valid_moves

    
    def clone(self):
        """Create a deep copy of the board"""
        new_board = Board(self.width, self.height)
        new_board.apple_grid = self.apple_grid.copy()
        new_board.score = self.score
        new_board.move_count = self.move_count
        return new_board
    
    def get_state(self):
        """Return the board state as a tensor for the neural network"""
        return self.apple_grid.copy()

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this board state
        self.children = []
        self.visits = 0
        self.value = 0  # Will now represent the board's average value
        self.untried_moves = self.get_untried_moves()
        
    def get_untried_moves(self):
        """Get list of moves not yet tried from this node"""
        return self.board.get_valid_moves()
    
    def select_child(self, exploration_weight=1.0):
        """Select a child node using UCB1 formula"""
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def ucb(child):
            # Lower board average is better, so we minimize
            # board_value = np.mean(child.board.apple_grid)
            board_value = len(child.board.get_valid_moves())
            
            # UCB1 formula: exploitation (minimizing) + exploration
            exploitation = -board_value
            exploration = exploration_weight * math.sqrt(log_visits / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration
        
        return max(self.children, key=ucb)
    
    def expand(self):
        """Expand the tree by adding a new child node"""
        if not self.untried_moves:
            return None
        
        # Choose a random untried move
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        # Create a new child node
        child_board = copy.deepcopy(self.board)
        child_board.move(move[0], move[1])
        child = MCTSNode(child_board, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.value += result
    
    def is_terminal(self):
        """Check if this node represents a terminal state"""
        return self.board.isClear() or not self.get_untried_moves()
    
    def is_fully_expanded(self):
        """Check if all possible child nodes have been expanded"""
        return len(self.untried_moves) == 0


class MCTS:
    def __init__(self, board, iterations=1000, exploration_weight=1):
        self.root = MCTSNode(board)
        self.iterations = iterations
        self.exploration_weight = exploration_weight
    
    def best_move(self):
        """Run MCTS and return the best move"""
        for _ in range(self.iterations):
            # Selection
            node = self.root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.exploration_weight)
            
            # Expansion
            if not node.is_terminal():
                node = node.expand()
                if node is None:  # No more moves to try
                    continue
            
            # Simulation
            board_copy = copy.deepcopy(node.board)
            result = self.simulate(board_copy)
            
            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent
        
        # Return the move with the lowest average board value
        if not self.root.children:
            return None
        
        def score(child):
            # If no cells left, return a perfect score
            if child.board.isClear():
                return float('inf')
            
            # Lower average board value is better
            # return -(np.mean(child.board.apple_grid) - 6.05) / 0.035
            return len(child.board.get_valid_moves())
        
        return max(self.root.children, key=score).move
    
    def simulate(self, board):
        """Run a simulation from the current board state."""
        depth = 0
        max_depth = 4  # Prevent infinite loops

        while depth < max_depth:
            valid_moves = board.get_valid_moves()
            if not valid_moves or board.isClear():
                break

            # Heuristic: //choose move that minimizes average board value
            # Heuristic: // new: choose move that maximizes number of possible moves + depth (to remove start/end bias).
            best_move = None
            # best_avg = float('inf')
            best_num_moves = float('-inf')

            for move in valid_moves:
                board_copy = copy.deepcopy(board)
                board_copy.move(move[0], move[1])

                # Calculate new average including zero cells
                # new_avg = np.mean(board_copy.apple_grid)
                new_num_moves = len(board_copy.get_valid_moves()) - depth - 1

                # if new_avg < best_avg:
                #     best_avg = new_avg
                #     best_move = move
                if new_num_moves > best_num_moves:
                    best_num_moves = new_num_moves
                    best_move = move

            if best_move is None:
                best_move = random.choice(valid_moves)  # Default to a random move

            board.move(best_move[0], best_move[1])
            depth += 1

        # If we reach a terminal state, use the final score as the heuristic
        num_moves_remaining = board.get_valid_moves()
        if board.isClear() or not num_moves_remaining:
            return 0

        # Otherwise, fall back to the average value heuristic
        # print(-(np.mean(board.apple_grid) - 6.05) / 0.035)
        # return -(np.mean(board.apple_grid) - 6.05) / 0.035
        return len(num_moves_remaining) - depth



def play_game(board, mcts_iterations=30, exploration_weight=1):
    moves_made = 0
    move_stack = []
    while True:
        print('depth', moves_made, end=' score ')
        mcts = MCTS(board, iterations=mcts_iterations, exploration_weight=exploration_weight)
        move = mcts.best_move()
        print(board.score)
        
        if move is None:
            break

        board.print_board()
        move_stack.append((move[0], move[1]))
        board.move(move[0], move[1])
        moves_made += 1
           
    board.print_board()
    return board.score, moves_made, move_stack


import time
# Sim mode (generate random board).
while True:
    start_time = time.time()
    board = Board()

    # Solve game.
    score, moves, move_stack = play_game(board, mcts_iterations=8, exploration_weight=.9)
    break

# # Real solve mode. (put game at 200% zoom, and align to top-left edge.)
# import mouse
# import image as img
# while True:
#     # press start button
#     start_time = time.time()
#     mouse.move(219, 1008, absolute=True, duration=0.1)
#     mouse.click("left")
#     mouse.move(477, 642, absolute=True, duration=0.1)
#     mouse.click("left")
#     time.sleep(0.2)
#     # capture the screen and read board state.
#     grid, offset = img.get_board()

#     board = Board(grid=grid)

#     moves = len(board.get_valid_moves())
#     mean = np.mean(board.apple_grid)

#     # Solve game.
#     score, moves, move_stack = play_game(board, mcts_iterations=8, exploration_weight=.9)
    
#     print(f"Final Score: {score}, Moves: {moves}, Average: {-np.mean(board.apple_grid)}")

#     for move in move_stack:
#         pos1, pos2 = move
#         x_1 = pos1[1]*66+offset[0]-20
#         y_1 = pos1[0]*66+offset[1]-20
#         x_2 = pos2[1]*66+offset[0]+60
#         y_2 = pos2[0]*66+offset[1]+60
#         mouse.drag(x_1, y_1, x_2, y_2, absolute=True, duration=.175)
