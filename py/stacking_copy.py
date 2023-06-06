
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

def blend_images_pyramid(images, masks, max_levels=3):
    # Ensure images and masks have the same length
    if len(images) != len(masks):
        raise ValueError('Images and masks lists must be of the same length')

    # Create Laplacian pyramid for each image and mask
    laplacian_pyramids_images = []
    gaussian_pyramids_masks = []
    for img, mask in zip(images, masks):
        # Create Gaussian pyramid for mask
        mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        gaussian_pyramid_mask = [mask]
        for _ in range(max_levels):
            gaussian_pyramid_mask.append(cv2.pyrDown(gaussian_pyramid_mask[-1]))
        gaussian_pyramids_masks.append(gaussian_pyramid_mask)

        # Create Laplacian pyramid for image
        img = img.astype(np.float32)
        gaussian_pyramid_img = [img]
        for _ in range(max_levels):
            gaussian_pyramid_img.append(cv2.pyrDown(gaussian_pyramid_img[-1]))
        laplacian_pyramid_img = [gaussian_pyramid_img[-1]]
        for i in range(max_levels, 0, -1):
            size = (gaussian_pyramid_img[i - 1].shape[1], gaussian_pyramid_img[i - 1].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid_img[i], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid_img[i - 1], expanded)
            laplacian_pyramid_img.append(laplacian)
        laplacian_pyramids_images.append(laplacian_pyramid_img[::-1])

    # Blend images
    blended_pyramids = []
    for i in range(max_levels + 1):
        # Apply mask to each channel individually
        blended_channels = []
        for channel in range(3):
            blended_channel = sum([cv2.multiply(laplacian_pyramids_images[j][i][:,:,channel], gaussian_pyramids_masks[j][i]) for j in range(len(images))])
            blended_channels.append(blended_channel)
        # Combine channels into single image
        blended = cv2.merge(blended_channels)
        blended = cv2.convertScaleAbs(blended)
        blended_pyramids.append(blended)

    # Reconstruct final image
    output_image = blended_pyramids[0]
    for i in range(1, max_levels + 1):
        output_image = cv2.pyrUp(output_image)
        if output_image.shape[:2] == blended_pyramids[i].shape[:2]:
            output_image = cv2.add(output_image, blended_pyramids[i])
        else:
            print(f"Skip level {i} due to size mismatch!")

    return output_image





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

# now with pyramid
res_img_pyramid = blend_images_pyramid(images, masks)

# Save the result
cv2.imwrite("result_pyramid.png", res_img_pyramid)