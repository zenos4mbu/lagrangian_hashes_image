import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

def load_image(image_path):
    """ Load an image from a file path. """
    return np.asarray(Image.open(image_path))

def generate_error_heatmap(original_image_path, reconstructed_image_path, sigma=1, vmin=None, vmax=None, output_path=None):
    # Load images
    original = load_image(original_image_path)
    reconstructed = load_image(reconstructed_image_path)

    # Calculate the difference
    error = np.abs(original.astype('float32') - reconstructed.astype('float32'))

    # Apply Gaussian smoothing
    smoothed_error = gaussian_filter(error, sigma=sigma)

    # Define a custom colormap
    colors = ["blue", "lightblue", "yellow"]  # Low to high error colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Generate heatmap
    plt.imshow(smoothed_error, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Display or save the heatmap
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


# Example usage
generate_error_heatmap('_results/logs/runs/1117_SPLASH_per_modifica_14/20231117-182628/img_gts400.png','_results/logs/runs/1117_SPLASH_per_modifica_14_PLUTO/20231117-195125/img_pred500.png',   sigma=2, vmin=0, vmax=0.01, output_path='error_PLUTO_14.png')
