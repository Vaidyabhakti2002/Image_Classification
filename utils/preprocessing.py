"""
Image preprocessing utilities
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


def resize_image(image, target_size=(224, 224)):
    """
    Resize image to target size
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size (height, width)
        
    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        return cv2.resize(image, target_size[::-1])
    elif isinstance(image, Image.Image):
        return image.resize(target_size[::-1])
    else:
        raise ValueError("Image must be numpy array or PIL Image")


def normalize_image(image, method='standard'):
    """
    Normalize image pixel values
    
    Args:
        image: Input image
        method: Normalization method ('standard', 'minmax', 'imagenet')
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    
    if method == 'standard':
        # Scale to [0, 1]
        return image / 255.0
    
    elif method == 'minmax':
        # Scale to [-1, 1]
        return (image / 127.5) - 1.0
    
    elif method == 'imagenet':
        # ImageNet normalization
        mean = np.array([123.68, 116.779, 103.939])
        return image - mean
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def augment_image(image, rotation_range=20, width_shift=0.2, height_shift=0.2,
                 horizontal_flip=True, zoom_range=0.2):
    """
    Apply data augmentation to image
    
    Args:
        image: Input image
        rotation_range: Range of random rotation in degrees
        width_shift: Fraction of width to shift
        height_shift: Fraction of height to shift
        horizontal_flip: Whether to randomly flip horizontally
        zoom_range: Range of random zoom
        
    Returns:
        Augmented image
    """
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random shift
    if width_shift > 0 or height_shift > 0:
        h, w = image.shape[:2]
        tx = np.random.uniform(-width_shift, width_shift) * w
        ty = np.random.uniform(-height_shift, height_shift) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random zoom
    if zoom_range > 0:
        zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        image = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = cv2.copyMakeBorder(image, pad_h, h-new_h-pad_h, 
                                      pad_w, w-new_w-pad_w, 
                                      cv2.BORDER_REFLECT)
    
    return image


def convert_to_grayscale(image):
    """
    Convert image to grayscale
    
    Args:
        image: Input image
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def convert_to_rgb(image):
    """
    Convert image to RGB
    
    Args:
        image: Input image
        
    Returns:
        RGB image
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


def apply_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        
    Returns:
        Adjusted image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image


def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian blur
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_edge_detection(image, method='canny'):
    """
    Apply edge detection
    
    Args:
        image: Input image
        method: Edge detection method ('canny', 'sobel', 'laplacian')
        
    Returns:
        Edge-detected image
    """
    gray = convert_to_grayscale(image)
    
    if method == 'canny':
        return cv2.Canny(gray, 100, 200)
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return np.sqrt(sobelx**2 + sobely**2)
    elif method == 'laplacian':
        return cv2.Laplacian(gray, cv2.CV_64F)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def crop_center(image, crop_size):
    """
    Crop center of image
    
    Args:
        image: Input image
        crop_size: Size of crop (height, width)
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[start_h:start_h+crop_h, start_w:start_w+crop_w]


def pad_image(image, target_size, mode='constant'):
    """
    Pad image to target size
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        mode: Padding mode ('constant', 'reflect', 'replicate')
        
    Returns:
        Padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    if mode == 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif mode == 'reflect':
        border_mode = cv2.BORDER_REFLECT
    elif mode == 'replicate':
        border_mode = cv2.BORDER_REPLICATE
    else:
        raise ValueError(f"Unknown padding mode: {mode}")
    
    return cv2.copyMakeBorder(image, top, bottom, left, right, border_mode)


def random_crop(image, crop_size):
    """
    Random crop from image
    
    Args:
        image: Input image
        crop_size: Size of crop (height, width)
        
    Returns:
        Randomly cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        raise ValueError("Crop size larger than image size")
    
    start_h = np.random.randint(0, h - crop_h + 1)
    start_w = np.random.randint(0, w - crop_w + 1)
    
    return image[start_h:start_h+crop_h, start_w:start_w+crop_w]


def mixup(images1, labels1, images2, labels2, alpha=0.2):
    """
    Apply mixup data augmentation
    
    Args:
        images1: First batch of images
        labels1: First batch of labels
        images2: Second batch of images
        labels2: Second batch of labels
        alpha: Mixup parameter
        
    Returns:
        Mixed images and labels
    """
    lam = np.random.beta(alpha, alpha)
    mixed_images = lam * images1 + (1 - lam) * images2
    mixed_labels = lam * labels1 + (1 - lam) * labels2
    
    return mixed_images, mixed_labels


def cutout(image, mask_size=16, n_masks=1):
    """
    Apply cutout augmentation
    
    Args:
        image: Input image
        mask_size: Size of cutout mask
        n_masks: Number of masks to apply
        
    Returns:
        Image with cutout
    """
    h, w = image.shape[:2]
    image = image.copy()
    
    for _ in range(n_masks):
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - mask_size // 2, 0, h)
        y2 = np.clip(y + mask_size // 2, 0, h)
        x1 = np.clip(x - mask_size // 2, 0, w)
        x2 = np.clip(x + mask_size // 2, 0, w)
        
        image[y1:y2, x1:x2] = 0
    
    return image


if __name__ == "__main__":
    print("Preprocessing utilities loaded successfully!")

# Made with Bob
