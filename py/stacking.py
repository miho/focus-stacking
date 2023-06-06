
import cv2
import numpy as np
import os
import glob
import pywt

def calculate_sharpness_pyr(image):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    depth = 2

    # Construct a Gaussian pyramid
    pyramid = [grayscale]
    for i in range(depth):
        pyramid.append(cv2.pyrDown(pyramid[i]))
    
    # Apply Laplacian to the smallest image in the pyramid
    laplacian = cv2.Laplacian(pyramid[-1], cv2.CV_64F)
    
    # Upscale the laplacian back to the original image size
    for i in range(depth, 0, -1):
        laplacian = cv2.pyrUp(laplacian)
        laplacian = cv2.resize(laplacian, (pyramid[i-1].shape[1], pyramid[i-1].shape[0]))
    
    # Calculate the absolute values to ignore edge direction
    laplacian = np.abs(laplacian)
    
    return laplacian

def calculate_sharpness_lap(image):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    # Compute the Laplacian of the image and then the Laplacian variance
    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)
    sharpness = cv2.convertScaleAbs(laplacian)
    
    return sharpness

def calculate_sharpness_ok(image):
    # wavelet transform
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grayscale = cv2.resize(grayscale, (image.shape[1]*2, image.shape[0]*2), interpolation = cv2.INTER_CUBIC)

    # Decompose the image using wavelet transform
    coeffs = pywt.dwt2(grayscale, 'db1')
    cA, (cH, cV, cD) = coeffs

    # Compute the energy of the high-frequency coefficients
    sharpness = np.sqrt(cH**2 + cV**2 + cD**2)

    # Increase contrast via histogram equalization
    sharpness = cv2.equalizeHist(sharpness.astype(np.uint8))

    return sharpness

# def calculate_sharpness(image):
#     # Convert the image to grayscale
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Decompose the image using wavelet transform
#     coeffs = pywt.dwt2(grayscale, 'db1')
#     cA, (cH, cV, cD) = coeffs

#     # Compute the energy of the high-frequency coefficients
#     energy = np.sqrt(cH**2 + cV**2 + cD**2)

#     # Integrate over a local neighborhood
#     kernel = np.ones((3, 3), np.float32)/9
#     sharpness = cv2.filter2D(energy, -1, kernel)

#     # Upscale the image to original size
#     sharpness = cv2.resize(sharpness, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_CUBIC)


#     # Increase contrast via histogram equalization
#     sharpness = cv2.equalizeHist(sharpness.astype(np.uint8))


#     return sharpness

def calculate_sharpness(image, max_levels=5):
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

    # # now increase contrast
    # sharpness_mask = cv2.equalizeHist(sharpness_mask)

    # Normalize to 0-1, then convert to 8-bit (0-255)
    #sharpness_mask = cv2.normalize(sharpness_mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Normalize to 8-bit (0-255), then convert to type np.uint8
    #sharpness_mask = cv2.normalize(sharpness_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    masks.append(sharpness_mask)

    # Write the sharpness mask to a new file
    cv2.imwrite(f"masks/sharpness_mask_{i}.png", sharpness_mask)


# blend images
res_img = blend_images(images, masks)

# Save the result
cv2.imwrite("result.png", res_img)