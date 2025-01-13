# Import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2

# Define a dictionary that maps the indexes of the facial landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4, 5))  # Adjusted range for nose
])

# In order to support legacy code, we'll default the indexes to the 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

def rect_to_bb(rect):
    """
    Convert dlib's rectangle object to (x, y, w, h) for OpenCV.
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    """
    Convert dlib's shape object (facial landmarks) to a NumPy array.
    """
    num_parts = shape.num_parts
    coords = np.zeros((num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to (x, y)-coordinates
    for i in range(0, num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def visualize_facial_landmarks(image, shape, model="68", colors=None, alpha=0.75):
    """
    Visualize facial landmarks on the given image.
    
    Parameters:
    - image: The input image.
    - shape: The coordinates of facial landmarks.
    - model: Either "68" or "5", to specify the landmark model.
    - colors: A list of colors for each facial region. If None, default colors are used.
    - alpha: Transparency factor for the overlay.
    
    Returns:
    - output: The image with facial landmarks drawn.
    """
    # Create two copies of the input image: one for overlay and one for output
    overlay = image.copy()
    output = image.copy()

    # Determine the facial landmark index map based on the model used
    if model == "68":
        landmark_idxs = FACIAL_LANDMARKS_68_IDXS
    elif model == "5":
        landmark_idxs = FACIAL_LANDMARKS_5_IDXS
    else:
        raise ValueError("Invalid model. Choose '68' or '5'.")

    # Initialize colors if none provided
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)]

    # Loop over each facial landmark region
    for (i, name) in enumerate(landmark_idxs.keys()):
        # Get the (x, y)-coordinates for the facial landmark region
        idxs = landmark_idxs[name]
        pts = shape[idxs[0]:idxs[1]] if len(idxs) == 2 else shape[idxs[0]:idxs[0] + 1]

        # Draw the jawline differently since it's not an enclosed region
        if name == "jaw" and model == "68":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i % len(colors)], 2)
        else:
            # Draw convex hull for enclosed regions like eyes, eyebrows, etc.
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i % len(colors)], -1)

    # Apply the overlay with transparency
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output
