import numpy as np
import cv2
import os
from skimage.color import label2rgb
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.color import label2rgb

def adaptive_slic(image, n_segments=2000, compactness=10, resize=(256,256), 
                  adaptive_compactness=True, display=2, return_type ='image'):
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
        plt.title('Adaptive Superpixels', color='white')
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
        plt.title('Adaptive Superpixels', color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    if return_type =='image':
        return superpixels_rgb
    # elif ret=='label':
    else:
        return superpixels
    


def save_images(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)

    image_name = os.path.basename(filename)
    directory = os.path.join(folder, image_name)
    cv2.imwrite(directory, image)
    return

def display_image(image, original=None, show_original=False):
    if show_original:
        plt.figure(figsize=(10,20),facecolor='black')

        plt.subplot(1,2,1)
        plt.imshow(original)
        plt.title('Original Image', color='white')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(image)
        plt.title('Adaptive Superpixels', color='white')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(10,10),facecolor='black', edgecolor='black')
        plt.imshow(image)
        plt.title('Modified', color='white')
        plt.axis('off')
        plt.show()

def show_contours_only(image, contours, display=False):
    # Create a black background
    black_background = np.zeros_like(image)

    # Draw contours on the black background
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), 3)

    if display:
        # Display the black background with white contours
        plt.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
        plt.title('Black Background')
        plt.axis('off')  # Hide the axis
        plt.show()

    return cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB)


def drawbb(image, contours, display=True):
    dimensions=[]
    image_copy=image.copy()
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw bounding box (optional)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 10)
        
        # Calculate dimensions and add to list
        dimensions.append((w, h))

    if display:
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title('BB')
        plt.axis('off')  # Hide the axis
        plt.show()

    print("Dimensions:", dimensions)


def filtered_contours(img, display=False):
    image=img.copy()
    cordnt_list = []
    # kernel = np.ones((5, 5), np.uint8) 

    t_lower = 50  # Lower Threshold 
    t_upper = 150  # Upper threshold 
  
# Applying the Canny Edge filter 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, t_lower, t_upper) 
    b_kernel=(3,3)
    blur=cv2.GaussianBlur(edge,b_kernel,0)
    # dilated = cv2.dilate(blur,None, iterations=2)
    # eroded = cv2.erode(dilated,None,iterations=1)
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)  
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 4000]

    image_copy=image.copy()

    if display:

        cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), 3) 
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title('filter by area')
        plt.axis('off')  # Hide the axis
        plt.show()

    return filtered_contours 