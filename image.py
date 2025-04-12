import cv2
import numpy as np
import pyautogui
import os
import numpy as np

def find_number_on_screen(number, screenshot, templates, threshold=0.97):
    """Finds occurrences of a number (1-9) in the given screenshot and returns a list of top-left positions."""

    # Get the template from preloaded dictionary
    template = templates.get(number)
    if template is None:
        return []  # Return empty list if template is missing

    # Perform template matching
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    
    # Get matching locations efficiently
    locations = np.column_stack(np.where(result >= threshold))
    
    # Extract (x, y) tuples
    positions = [(x, y) for y, x in locations]  # OpenCV returns (row, col), so swap
    
    return positions


def get_board():
    """Detects numbers on the screen, builds a board representation, and returns (board, board_offset)."""
    
    # Capture screenshot once
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot, dtype=np.uint8)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    # Preload number templates
    templates = {}
    for i in range(1, 10):
        file_name = f"./res/{i}.png"
        if os.path.exists(file_name):
            template = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            
            # Remove alpha channel if present
            if template.shape[-1] == 4:
                template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
            
            templates[i] = template

    # Dictionary to store found numbers
    board_dict = {}
    
    for i in range(1, 10):
        positions = find_number_on_screen(i, screenshot, templates)
        positions = [pos for pos in positions if pos[1] <= 1000]  # Filter out y > 1000
        if positions:
            board_dict[i] = positions

    if not board_dict:
        return None, None  # No numbers found
    
    # Find board_offset (smallest x, y)
    min_x = min(pos[0] for positions in board_dict.values() for pos in positions)
    min_y = min(pos[1] for positions in board_dict.values() for pos in positions)
    board_offset = (min_x, min_y)

    # Initialize board as a NumPy array
    grid_size_x = 17
    grid_size_y = 10
    board = np.zeros((grid_size_y, grid_size_x), dtype=np.int8)


    # Fill board based on positions
    for number, positions in board_dict.items():
        for x, y in positions:
            i = round((x - board_offset[0]) / 66)
            j = round((y - board_offset[1]) / 66)
            if 0 <= j < grid_size_y and 0 <= i < grid_size_x:  # Ensure valid indices
                board[j, i] = number  # NumPy uses (row, col) indexing

    return board, board_offset


# # Run the function and print board
# board, board_offset = get_board()
# if board is not None:
#     for row in board:
#         print(row)
#     print("Board Offset:", board_offset)
# else:
#     print("No numbers detected.")
