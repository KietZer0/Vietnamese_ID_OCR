# utils.py
import cv2
import numpy as np

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def display_image(window_name, image, wait_key=0):
    """Displays an image using OpenCV or Colab's cv2_imshow."""
    if IN_COLAB:
        # Convert BGR to RGB for Colab display if needed
        # if image.shape[2] == 3:
        #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # else:
        #     image_rgb = image
        cv2_imshow(image) # cv2_imshow handles BGR correctly
    else:
        cv2.imshow(window_name, image)
        key_pressed = cv2.waitKey(wait_key)
        if wait_key == 0 or key_pressed != -1: # Destroy window if key pressed or indefinite wait
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                # Window might have been closed manually or not exist
                pass

def order_points(pts):
    """Orders a list of 4 points for perspective transform:
    top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    """Applies a perspective transform to an image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped

def preprocess_for_ocr(image_roi):
    """Basic preprocessing for an image ROI before OCR."""
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    # Apply thresholding - adaptive might be better for varying lighting
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # For now, let's try a simple binary threshold or even just grayscale
    # Tesseract prefers dark text on light background. If your ROI is inverted, invert it.
    # This might need adjustment based on how ROIs are extracted.
    # For now, assume ROIs are fine as is or grayscale is enough.
    # Example: _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # To make text black on white:
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # if np.mean(thresh) > 127: # if most of the image is white, invert
    # thresh = cv2.bitwise_not(thresh)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Rescale for potentially better OCR (Tesseract likes ~300 DPI)
    # target_height = 60 #pixels, adjust as needed
    # current_height = denoised.shape[0]
    # if current_height > 0 and current_height < target_height * 0.8 : # Only scale up if significantly smaller
    #     scale_factor = target_height / current_height
    #     rescaled = cv2.resize(denoised, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    #     return rescaled
    return denoised