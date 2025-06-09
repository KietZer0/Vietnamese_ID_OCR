# card_preprocessor.py
import cv2
import numpy as np
from .utils import perspective_transform, display_image # Assuming utils.py is in the same directory or package

def find_card_contour(image, debug=False):
    """Finds the largest rectangular contour, assumed to be the ID card."""
    orig_height, orig_width = image.shape[:2]
    
    # Resize for faster processing, keeping aspect ratio
    ratio = orig_height / 500.0
    img_resized = cv2.resize(image, (int(orig_width / ratio), 500))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    if debug:
        display_image("Edged", edged)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    card_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_contour = approx
            break
    
    if card_contour is None:
        return None

    if debug:
        debug_img = img_resized.copy()
        cv2.drawContours(debug_img, [card_contour], -1, (0, 255, 0), 2)
        display_image("Card Contour on Resized", debug_img)

    # Scale contour back to original image size
    return (card_contour.reshape(4, 2) * ratio).astype(np.float32)


def crop_and_warp_card(image, debug=False):
    """Detects the ID card in an image and returns a warped, top-down view."""
    card_points_orig = find_card_contour(image, debug=debug)

    if card_points_orig is None:
        print("Could not find ID card contour.")
        return None

    warped_card = perspective_transform(image, card_points_orig)
    
    # Standardize card orientation (optional, but good for consistency)
    # Most ID cards are wider than they are tall
    h, w = warped_card.shape[:2]
    if h > w:
        warped_card = cv2.rotate(warped_card, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    if debug:
        display_image("Warped Card", cv2.resize(warped_card, (0,0), fx=0.5, fy=0.5))
        
    return warped_card

def extract_face_roi(warped_card_image, roi_factors):
    """Extracts the face ROI from the warped card image."""
    h, w = warped_card_image.shape[:2]
    ymin_f, xmin_f, ymax_f, xmax_f = roi_factors
    
    ymin = int(h * ymin_f)
    xmin = int(w * xmin_f)
    ymax = int(h * ymax_f)
    xmax = int(w * xmax_f)
    
    face_image = warped_card_image[ymin:ymax, xmin:xmax]
    return face_image