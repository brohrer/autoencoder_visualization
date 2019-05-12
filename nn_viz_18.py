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

    p = find_node_image_size(p)
    p = find_between_layer_gap(p)
    p = find_between_node_gap(p)
    p = find_error_image_position(p)

    add_input_image(fig, p)
    save_nn_viz(fig, postfix="18_input_random_refactored")


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


def find_node_image_size(p):
    """
    What should the height and width of each node image be?
    As big as possible, given the constraints.
    There are two possible constraints:
        1. Fill the figure top-to-bottom.
        2. Fill the figure side-to-side.
    To determine which of these limits the size of the node images,
    we'll calculate the image size assuming each constraint separately,
    then respect the one that results in the smaller node image.
    """
    # First assume height is the limiting factor.
    total_space_to_fill = (
        p["figure"]["height"]
        - p["gap"]["bottom_border"]
        - p["gap"]["top_border"]
    )
    # Use the layer with the largest number of nodes (n_max).
    # Pack the images and the gaps as tight as possible.
    # In that case, if the image height is h,
    # the gaps will each be h * p["gap"]["between_node_scale"].
    # There will be n_max nodes and (n_max - 1) gaps.
    # After a wee bit of algebra:
    height_constrained_by_height = (
        total_space_to_fill / (
           p["network"]["max_nodes"]
           + (p["network"]["max_nodes"] - 1)
           * p["gap"]["between_node_scale"]
        )
    )

    # Second assume width is the limiting factor.
    total_space_to_fill = (
        p["figure"]["width"]
        - p["gap"]["left_border"]
        - p["gap"]["right_border"]
        - 2 * p["input"]["image"]["width"]
    )
    # Again, pack the images as tightly as possible side-to-side.
    # In this case, if the image width is w,
    # the gaps will each be w * p["gap"]["between_layer_scale"].
    # There will be n_layer nodes and (n_layer + 1) gaps.
    # After another tidbit of algebra:
    width_constrained_by_width = (
        total_space_to_fill / (
           p["network"]["n_layers"]
           + (p["network"]["n_layers"] + 1)
           * p["gap"]["between_layer_scale"]
        )
    )

    # Figure out what the height would be for this width.
    height_constrained_by_width = (
        width_constrained_by_width
        / p["input"]["aspect_ratio"]
    )

    # See which constraint is more restrictive, and go with that one.
    p["node_image"]["height"] = np.minimum(
        height_constrained_by_width,
        height_constrained_by_height)
    p["node_image"]["width"] = (
        p["node_image"]["height"]
        * p["input"]["aspect_ratio"]
    )
    return p


def find_between_layer_gap(p):
    """
    How big is the horizontal spacing between_layers?
    This is also the spacing between the input image and the first layer
    and between the last layer and the output image.
    """
    horizontal_gap_total = (
        p["figure"]["width"]
        - 2 * p["input"]["image"]["width"]
        - p["network"]["n_layers"] * p["node_image"]["width"]
        - p["gap"]["left_border"]
        - p["gap"]["right_border"]
    )
    n_horizontal_gaps = p["network"]["n_layers"] + 1
    p["gap"]["between_layer"] = horizontal_gap_total / n_horizontal_gaps
    return p


def find_between_node_gap(p):
    """
    How big is the vertical gap between_node images?
    """
    vertical_gap_total = (
        p["figure"]["height"]
        - p["gap"]["top_border"]
        - p["gap"]["bottom_border"]
        - p["network"]["max_nodes"]
        * p["node_image"]["height"]
    )
    n_vertical_gaps = p["network"]["max_nodes"] - 1
    p["gap"]["between_node"] = vertical_gap_total / n_vertical_gaps
    return p


def find_error_image_position(p):
    """
    Where exactly should the error image be positioned?
    """
    p["error_image"]["bottom"] = (
        p["input"]["image"]["bottom"]
        - p["input"]["image"]["height"]
        * p["gap"]["error_gap_scale"]
        - p["error_image"]["height"]
    )
    error_image_center = (
        p["figure"]["width"]
        - p["gap"]["right_border"]
        - p["input"]["image"]["width"] / 2
    )
    p["error_image"]["left"] = (
        error_image_center
        - p["error_image"]["width"] / 2
    )
    return p


def add_input_image(fig, p):
    """
    All Axes to be added use the rectangle specification
        (left, bottom, width, height)
    """
    absolute_pos = (
        p["gap"]["left_border"],
        p["input"]["image"]["bottom"],
        p["input"]["image"]["width"],
        p["input"]["image"]["height"])
    ax_input = add_image_axes(fig, p, absolute_pos)
    fill_patch = np.random.sample(size=(
       p["input"]["n_rows"],
       p["input"]["n_cols"],
    ))
    ax_input.imshow(fill_patch, cmap="inferno")


def add_image_axes(fig, p, absolute_pos):
    """
    Locate the Axes for the image corresponding to this node within the Figure.

    absolute_pos: Tuple of
        (left_position, bottom_position, width, height)
    in inches on the Figure.
    """
    scaled_pos = (
        absolute_pos[0] / p["figure"]["width"],
        absolute_pos[1] / p["figure"]["height"],
        absolute_pos[2] / p["figure"]["width"],
        absolute_pos[3] / p["figure"]["height"])
    ax = fig.add_axes(scaled_pos)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.spines["top"].set_color(TAN)
    ax.spines["bottom"].set_color(TAN)
    ax.spines["left"].set_color(TAN)
    ax.spines["right"].set_color(TAN)
    return ax


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
