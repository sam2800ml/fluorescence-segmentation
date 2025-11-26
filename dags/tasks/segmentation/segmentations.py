from skimage import (
    io, color, filters, morphology, measure, exposure, segmentation
)
from skimage.filters import (
    gaussian, threshold_otsu, threshold_local, threshold_multiotsu, sobel
)
from skimage.segmentation import clear_border, watershed
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
import numpy as np
from tasks.segmentation.preprocessing_images import preprocess_array


def segment_cytoplasm_with_nuclei(image, nuclei_mask):
    """Segmentation of the cytoplasm"""
    _, gray_norm = preprocess_array(image)
    im_blurred = filters.gaussian(gray_norm, sigma=2)

    # Base threshold for foreground
    binary_mask = im_blurred > filters.threshold_otsu(im_blurred)

    # Distance transform from cytoplasm foreground
    distance = ndi.distance_transform_edt(binary_mask)

    # Use nuclei as markers
    labels = measure.label(nuclei_mask)

    # Watershed: expand nuclei into cytoplasm
    cytoplasm_labels = segmentation.watershed(-distance, labels, mask=binary_mask)

    # Convert to binary (1 for cytoplasm region, 0 for background)
    cytoplasm_binary = (cytoplasm_labels > 0).astype(np.uint8)

    return cytoplasm_binary


def segment_nucleus(image):
    """Segmentation of the nucleus"""
    # Convert to grayscale
    _, gray_norm = preprocess_array(image)

    # Smooth to reduce noise
    im_blurred = filters.gaussian(gray_norm, sigma=1)

    # Threshold (Otsu for global separation)
    thresh = filters.threshold_otsu(im_blurred)
    binary_mask = im_blurred > thresh   # Cytoplasm should appear as True region

    # Morphological cleaning
    mask = morphology.remove_small_objects(binary_mask, min_size=400)
    mask = morphology.remove_small_holes(mask, area_threshold=1000)
    mask = morphology.opening(mask, morphology.disk(3))

    # Fill internal holes
    mask = ndi.binary_fill_holes(mask)
    return mask


def segment_protein(image,grad_threshold=0.1, disk_radius=3):
    """Segmentation of the protein"""
    _, gray_norm = preprocess_array(image)

    # Multi-Otsu segmentation

    im_blurred = filters.gaussian(gray_norm, sigma=1)
    thresholds = threshold_multiotsu(gray_norm, classes=3)
    regions = np.digitize(gray_norm, bins=thresholds)
        # Select the brightest region (like mask_otsu = regions == 2 in first script)
    mask_otsu = regions == 2

    # Gradient-based mask
    grad_mag = sobel(gray_norm)
    grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-10)
    grad_mask = grad_norm > grad_threshold

    # Combine intensity + edges
    combined = mask_otsu & grad_mask

    # Morphological cleaning
    cleaned = morphology.binary_closing(combined, morphology.disk(disk_radius))
    cleaned = ndi.binary_fill_holes(cleaned)
    cleaned = clear_border(cleaned)

    return cleaned
