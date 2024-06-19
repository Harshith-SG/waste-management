import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

def show_contours_only(image, contours):
    # Create a black background
    black_background = np.zeros_like(image)

    # Draw contours on the black background
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), 3)

    # Display the black background with white contours
    plt.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis
    plt.show()
def img_to_polygons(image):
    cordnt_list = []
    kernel = np.ones((5, 5), np.uint8) 
    image = cv2.dilate(image, kernel, iterations=1)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grabcontours(contours)
    print("Number of Contours found = " + str(len(contours))) 
    contours=list(contours)
    image_copy=image.copy()
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3) 
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis
    plt.show()
    
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 2000]
    image_copy=image.copy()

    print("Number of Contours found = " + str(len(filtered_contours))) 
    cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), 3) 
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis
    plt.show()

    image_copy=image.copy()
    cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), 3) 
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis
    plt.show()

    show_contours_only(image, filtered_contours)
    # # Extract coordinates of white regions
    # white_regions_coordinates = []
    # for contour in contours:
    #     coordinates = contour.squeeze().astype(float).tolist()
    #     white_regions_coordinates.append(coordinates)

    # # Print the coordinates of each white region
    # for i, region in enumerate(white_regions_coordinates):
    #     if type(region[0]) is list:
    #         if len(region) > 2:
    #             # Calculate the area of the contour
    #             area = cv2.contourArea(np.around(np.array([[pnt] for pnt in region])).astype(np.int32))
    #             if area > 100:
    #                 # Convert coordinates to the required format
    #                 crdnts = [{'x': i[0], 'y': i[1]} for i in region]
    #                 cordnt_list.append(crdnts)

    # return cordnt_list
    return contours

# Set the current working directory
image_dir="C:\waste-management\train_images\IMG-20240618-WA0014.jpg"
image=cv2.imread('2.jpeg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide the axis
plt.show()
# cv2.imshow('iamge',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img_to_polygons(image)