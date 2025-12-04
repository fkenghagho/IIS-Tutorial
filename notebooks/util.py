import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from skimage.draw import disk, rectangle, circle_perimeter
from skimage.transform import rotate
from skimage.util import img_as_ubyte
from PIL import Image, ImageFilter, ImageOps
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy.interpolate import CubicSpline
import random


def display_image_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    figsize: tuple = (15, 15),
    cmap: str = 'gray',
    title_prefix: str = ""
):
    """
    Displays a list of images in a grid layout, suitable for Jupyter Notebooks.

    This function is excellent for visualizing the knowledge base (kb) images,
    comparing test images, or showing the output of an algorithm like Edge/SIFT.

    Args:
        images (List[np.ndarray]): A list of numpy arrays representing the images.
                                   Supports 2D (grayscale) and 3D (RGB/BGR) arrays.
        titles (Optional[List[str]]): An optional list of titles for each image.
                                      If None, no titles are displayed.
        cols (int): The number of columns in the grid.
        figsize (tuple): The size (width, height) of the overall figure in inches.
        cmap (str): The colormap to use for 2D (grayscale) images. Defaults to 'gray'.
        title_prefix (str): A prefix to add to all individual image titles.
    """
    # Calculate the number of rows needed
    n_images = len(images)
    if n_images == 0:
        print("Error: No images provided to display.")
        return

    rows = int(np.ceil(n_images / cols))
    
    # Create the figure and subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten the axes array for easy iteration, handling cases of 1 row or 1 column
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Iterate over the images and plot them
    for i, img in enumerate(images):
        ax = axes[i]
        
        # Determine the correct colormap based on image dimensions
        # If the image is 3D (e.g., color), set cmap to None to use default color handling
        current_cmap = cmap if img.ndim == 2 else None
        
        # Display the image
        ax.imshow(img, cmap=current_cmap)
        
        # Set the title if provided
        if titles and i < len(titles):
            ax.set_title(f"{title_prefix}{titles[i]}", fontsize=10)
        
        # Remove axis ticks and labels for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(n_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- Example Usage (Demonstration) ---
    
    # 1. Create 6 distinct synthetic images (simulating sensor data or training set)
    SIZE = 50
    images = []
    titles = []
    
    # Image 1: Simple white square
    img1 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    img1[10:40, 10:40] = 255
    images.append(img1); titles.append("White Square (Gray)")

    # Image 2: Gradient color (simulated RGB input)
    img2 = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(SIZE):
        img2[i, :, 0] = int(255 * (i / SIZE))  # Red gradient
        img2[:, i, 1] = int(255 * (i / SIZE))  # Green gradient
    images.append(img2); titles.append("RGB Gradient")

    # Image 3: Diagonal line (edge feature test)
    img3 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    np.fill_diagonal(img3, 200)
    images.append(img3); titles.append("Diagonal Edge")
    
    # Image 4: Random noise (sonar/depth data visualization)
    img4 = np.random.randint(0, 256, (SIZE, SIZE), dtype=np.uint8)
    images.append(img4); titles.append("Random Noise")
    
    # Image 5: A custom colormap (depth visualization example)
    img5 = np.arange(SIZE*SIZE).reshape(SIZE, SIZE)
    images.append(img5); titles.append("Depth Map (viridis)")

    # Image 6: Another colored image (BGR format)
    img6 = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img6[:, :, 2] = 150 # Blue channel
    img6[20:30, 20:30, 0] = 255 # Red spot
    images.append(img6); titles.append("BGR Image")
    
    # 2. Call the display function
    print("Displaying 6 images in a 3x2 grid:")
    display_image_grid(
        images, 
        titles=titles, 
        cols=3, 
        figsize=(10, 7), 
        cmap='viridis', # Use a specific colormap for grayscale images
        title_prefix="Fig "
    )
    
    print("\nDisplaying the first 4 images in a 2x2 grid:")
    display_image_grid(
        images[:4], 
        titles=titles[:4], 
        cols=2, 
        figsize=(8, 8), 
        cmap='plasma'
    )