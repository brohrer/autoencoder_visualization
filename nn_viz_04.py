"""
Generate an autoencoder neural network visualization
"""
import numpy as np

# Changing these adjusts the size and layout of the visualization
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 9
RIGHT_BORDER = 0.7
LEFT_BORDER = 0.7
TOP_BORDER = 0.8
BOTTOM_BORDER = 0.6

N_IMAGE_PIXEL_COLS = 64
N_IMAGE_PIXEL_ROWS = 48
N_NODES_BY_LAYER = [10, 7, 5, 8]

INPUT_IMAGE_BOTTOM = 5
INPUT_IMAGE_HEIGHT = 0.25 * FIGURE_HEIGHT
ERROR_IMAGE_SCALE = 0.7
ERROR_GAP_SCALE = 0.3
BETWEEN_LAYER_SCALE = 0.8
BETWEEN_NODE_SCALE = 0.4


def main():
    """
    Build a visualization of an image autoencoder neural network,
    piece by piece.

    A central data structure in this example is the collection of parameters
    that define how the image is laid out. It is a set of nested dictionaries.
    """
    p = construct_parameters()
    print("parameters:")
    for key, value in p.items():
        print(key, ":", value)


def construct_parameters():
    """
    Build a dictionary of parameters that describe the size and location
    of the elements of the visualization. This is a convenient way to pass
    the collection of them around .
    """
    # Enforce square pixels. Each pixel will have the same height and width.
    aspect_ratio = N_IMAGE_PIXEL_COLS / N_IMAGE_PIXEL_ROWS

    parameters = {}

    # The figure as a whole
    parameters["figure"] = {
        "height": FIGURE_HEIGHT,
        "width": FIGURE_WIDTH,
    }

    # The input and output images
    parameters["input"] = {
        "n_cols": N_IMAGE_PIXEL_COLS,
        "n_rows": N_IMAGE_PIXEL_ROWS,
        "aspect_ratio": aspect_ratio,
        "image": {
            "bottom": INPUT_IMAGE_BOTTOM,
            "height": INPUT_IMAGE_HEIGHT,
            "width": INPUT_IMAGE_HEIGHT * aspect_ratio,
        }
    }

    # The network as a whole
    parameters["network"] = {
        "n_nodes": N_NODES_BY_LAYER,
        "n_layers": len(N_NODES_BY_LAYER),
        "max_nodes": np.max(N_NODES_BY_LAYER),
    }

    # Individual node images
    parameters["node_image"] = {
       "height": 0,
       "width": 0,
    }

    return parameters


if __name__ == "__main__":
    main()
