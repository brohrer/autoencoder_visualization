"""
Generate an autoencoder neural network visualization
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Choose a color palette
BLUE = "#04253a"
GREEN = "#4c837a"
TAN = "#e1ddbf"
DPI = 300

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
    fig, ax_boss = create_background(p)
    save_nn_viz(fig, postfix="08_hi_res")


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

    parameters["error_image"] = {
        "left": 0,
        "bottom": 0,
        "width": parameters["input"]["image"]["width"] * ERROR_IMAGE_SCALE,
        "height": parameters["input"]["image"]["height"] * ERROR_IMAGE_SCALE,
    }

    parameters["gap"] = {
       "right_border": RIGHT_BORDER,
       "left_border": LEFT_BORDER,
       "bottom_border": BOTTOM_BORDER,
       "top_border": TOP_BORDER,
       "between_layer": 0,
       "between_layer_scale": BETWEEN_LAYER_SCALE,
       "between_node": 0,
       "between_node_scale": BETWEEN_NODE_SCALE,
       "error_gap_scale": ERROR_GAP_SCALE,
    }

    return parameters


def create_background(p):
    fig = plt.figure(
        edgecolor=TAN,
        facecolor=GREEN,
        figsize=(p["figure"]["width"], p["figure"]["height"]),
        linewidth=4,
    )
    ax_boss = fig.add_axes((0, 0, 1, 1), facecolor="none")
    ax_boss.set_xlim(0, 1)
    ax_boss.set_ylim(0, 1)
    return fig, ax_boss


def save_nn_viz(fig, postfix="0"):
    """
    Generate a new filename for each step of the process.
    """
    base_name = "nn_viz_"
    filename = base_name + postfix + ".png"
    fig.savefig(
        filename,
        edgecolor=fig.get_edgecolor(),
        facecolor=fig.get_facecolor(),
        dpi=DPI,
    )


if __name__ == "__main__":
    main()
