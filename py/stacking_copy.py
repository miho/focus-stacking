
import cv2
import numpy as np
import os
import glob
import pywt

def calculate_sharpness(image, max_levels=3):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Construct the Laplacian pyramid
    pyramid = [grayscale]
    for i in range(max_levels):
        next_level = cv2.pyrDown(pyramid[-1])
        pyramid.append(next_level)

    # For each level in the pyramid, calculate the local energy
    energy_pyramid = []
    for level in pyramid:
        laplacian = cv2.Laplacian(level, cv2.CV_64F)
        energy = cv2.convertScaleAbs(laplacian)
        # Upscale to the original size
        energy = cv2.resize(energy, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
        energy_pyramid.append(energy)

    # Sum up the energy images to get the combined sharpness map
    sharpness = np.sum(energy_pyramid, axis=0)

    return sharpness


def gamma_correction(image, gamma):
    # Create a lookup table for the gamma correction
    lookup_table = np.array([((i/255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    
    # Apply the lookup table to the image
    corrected_image = cv2.LUT(image, lookup_table)
    
    return corrected_image


def blend_images(images, masks):
    # Ensure all masks are 3-channel, same as the images
    masks = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in masks]
    
    # Convert the masks so that their values range between 0 and 1
    masks = [mask.astype(float)/255 for mask in masks]

    # Sum of all masks
    mask_sum = np.sum(masks, axis=0)
    
    # Normalized masks: for each pixel, the sum of weights across all images will be 1
    norm_masks = [mask / (mask_sum + 1e-6) for mask in masks]  # the small value is to prevent division by zero

    # Multiply each image by its corresponding normalized mask and sum all
    blended = np.sum([img * mask for img, mask in zip(images, norm_masks)], axis=0)

    # Rescale to [0, 255] and convert to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

# Specify the source directory of the images
src_directory = "C:\\SP7-DATA\\Users\\miho\\Downloads\\focus-stacking-master\\focus-stacking-master\\example-images"    


# Create a new directory for the masks
os.makedirs("masks", exist_ok=True)

images = []
masks = []

# Load all the images in the folder
image_files = glob.glob(os.path.join(src_directory, "*"))
for i, img_file in enumerate(image_files):
    # Load the image
    img = cv2.imread(img_file)
    images.append(img)
    
    # Calculate the sharpness mask
    sharpness_mask = calculate_sharpness(img)

    # Percentile-based normalization to 0-1
    lower, upper = np.percentile(sharpness_mask, [1, 99])
    sharpness_mask = np.clip(sharpness_mask, lower, upper)
    sharpness_mask = (sharpness_mask - lower) / (upper - lower)
    # Convert to 8-bit (0-255)
    sharpness_mask = (255 * sharpness_mask).astype(np.uint8)

    sharpness_mask = gamma_correction(sharpness_mask, 25)

    # replace 0 with 1 to avoid division by zero
    sharpness_mask[sharpness_mask == 0] = 1

    masks.append(sharpness_mask)

    # Write the sharpness mask to a new file
    cv2.imwrite(f"masks/sharpness_mask_{i}.png", sharpness_mask)


# blend images
res_img = blend_images(images, masks)

# Save the result
cv2.imwrite("result.png", res_img)

