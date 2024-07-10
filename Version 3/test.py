import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.color import rgb2lab
from scipy.stats import multivariate_normal

def compute_superpixels(image, n_segments=100, compactness=10):
    lab_image = rgb2lab(image)
    segments = slic(lab_image, n_segments=n_segments, compactness=compactness, start_label=1)
    return segments

def compute_color_model(image, segments, n_segments):
    height, width, _ = image.shape
    lab_image = rgb2lab(image)

    superpixel_colors = []
    for segment_id in range(1, n_segments + 1):
        mask = (segments == segment_id)
        color_values = lab_image[mask]
        superpixel_colors.append(color_values)

    means = [np.mean(colors, axis=0) for colors in superpixel_colors]
    covariances = [np.cov(colors, rowvar=False) for colors in superpixel_colors]

    return means, covariances

def bayesian_segmentation(image, segments, means, covariances, n_segments):
    height, width, _ = image.shape
    lab_image = rgb2lab(image)
    bayesian_segments = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            pixel = lab_image[y, x]
            max_prob = -np.inf
            best_segment = -1

            for segment_id in range(n_segments):
                mean = means[segment_id]
                cov = covariances[segment_id]
                prob = multivariate_normal.pdf(pixel, mean=mean, cov=cov)

                if prob > max_prob:
                    max_prob = prob
                    best_segment = segment_id + 1

            bayesian_segments[y, x] = best_segment

    return bayesian_segments

def main(image_path, n_segments=100, compactness=10):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segments = compute_superpixels(image, n_segments, compactness)
    means, covariances = compute_color_model(image, segments, n_segments)
    bayesian_segments = bayesian_segmentation(image, segments, means, covariances, n_segments)

    return bayesian_segments

if __name__ == "__main__":
    image_path = 'train_images\IMG-20240618-WA0013.jpg'
    segmented_image = main(image_path)

    # Display the segmented image
    cv2.imshow("Segmented Image", segmented_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
