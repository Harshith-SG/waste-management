import numpy as np
import cv2
from skimage.color import label2rgb
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.color import label2rgb

def adaptive_slic(image, n_segments=2000, compactness=10, resize=(256,256), 
                  adaptive_compactness=True, display=2):
    img_resized = cv2.resize(image, resize)
    if adaptive_compactness:
        # Compute gradient magnitude
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gradient = cv2.Laplacian(gray, cv2.CV_64F)
        gradient_magnitude = np.sqrt(gradient ** 2)
        
        # Normalize gradient magnitude
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - 
                                                                                gradient_magnitude.min())
        
        # Adjust compactness based on gradient
        adaptive_compactness = compactness * (1 + gradient_magnitude)
        
        # Apply SLIC with adaptive compactness
        superpixels = slic(img_resized, n_segments=n_segments, compactness=adaptive_compactness.mean())
    else:
        # Apply standard SLIC
        superpixels = slic(img_resized, n_segments=n_segments, compactness=compactness)


    # Convert superpixel labels to an RGB image for visualization
    superpixels_rgb = label2rgb(superpixels, img_resized, kind='avg')

    if display==3:
        # Plot the superpixel segmentation
        plt.figure(figsize=(10, 30),facecolor='black', edgecolor='white')
        plt.subplot(1,3,1)
        plt.imshow(img_resized)
        plt.title('Original Image', color='white')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(gradient_magnitude, cmap='gist_heat')
        plt.title('Gradient', color='white')
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(superpixels_rgb)
        plt.title('Superpixel Segmentation', color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    if display==2:
        # Plot the superpixel segmentation
        plt.figure(figsize=(10, 20),facecolor='black', edgecolor='white')
        plt.subplot(1,2,1)
        plt.imshow(img_resized)
        plt.title('Original Image', color='white')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(superpixels_rgb)
        plt.title('Superpixel Segmentation', color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
