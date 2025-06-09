# card_preprocessor.py
import cv2
import numpy as np
from utils import perspective_transform, display_image # Assuming utils.py is in the same directory or package

def find_card_contour(image, debug=False):
    """Finds the largest rectangular contour, assumed to be the ID card."""
    orig_height, orig_width = image.shape[:2]
    
    # Resize for faster processing, keeping aspect ratio
    # Target height for processing
    processing_height = 600.0 # Increased from 500
    ratio = orig_height / processing_height
    img_resized = cv2.resize(image, (int(orig_width / ratio), int(processing_height)))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Experiment with different blurring
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75) # Better at preserving edges

    # Experiment with Canny edge detection parameters
    # Lower thresholds might pick up more edges, higher thresholds fewer.
    edged = cv2.Canny(blurred, 50, 150) # Adjusted from 75, 200

    if debug:
        display_image("Resized Gray", gray)
        display_image("Resized Blurred", blurred)
        display_image("Resized Edged", edged)

    # Morphological operations to close gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2) # iterations can be tuned

    if debug:
        display_image("Closed Edges", closed)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Use RETR_EXTERNAL
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # Check more contours

    card_contour_poly = None
    min_contour_area = (img_resized.shape[0] * img_resized.shape[1]) * 0.1 # Card should be at least 10% of image area

    for c in contours:
        if cv2.contourArea(c) < min_contour_area: # Filter out small contours early
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # Epsilon can be tuned (0.01 to 0.05 usually)
        
        if len(approx) == 4:
            # Further check if it's reasonably rectangular and has ID card aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            # Standard ID card aspect ratio is ~1.586 (85.6mm / 53.98mm)
            # Allow some tolerance
            if 1.3 < aspect_ratio < 1.9 or 1/1.9 < aspect_ratio < 1/1.3 : # Check both orientations
                card_contour_poly = approx
                if debug:
                    print(f"Found potential card contour with area {cv2.contourArea(c)} and aspect ratio {aspect_ratio}")
                break
    
    if card_contour_poly is None:
        if debug:
            print("No suitable 4-sided contour found.")
            # Draw all top contours for debugging
            debug_img_all_contours = img_resized.copy()
            cv2.drawContours(debug_img_all_contours, contours, -1, (0,0,255), 1)
            display_image("All Top Contours", debug_img_all_contours)
        return None

    if debug:
        debug_img_contour = img_resized.copy()
        cv2.drawContours(debug_img_contour, [card_contour_poly], -1, (0, 255, 0), 2)
        display_image("Card Contour on Resized", debug_img_contour)

    # Scale contour back to original image size
    return (card_contour_poly.reshape(4, 2) * ratio).astype(np.float32)


def crop_and_warp_card(image, debug=False):
    """Detects the ID card in an image and returns a warped, top-down view."""
    card_points_orig = find_card_contour(image, debug=debug)

    if card_points_orig is None:
        print("Could not find ID card contour in crop_and_warp_card.")
        return None

    warped_card = perspective_transform(image, card_points_orig)
    
    h, w = warped_card.shape[:2]
    if h == 0 or w == 0:
        print("Warped card has zero dimension.")
        return None
        
    # Standardize card orientation to landscape (wider than tall)
    if h > w:
        warped_card = cv2.rotate(warped_card, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    if debug:
        display_image("Warped Card (Final)", cv2.resize(warped_card, (0,0), fx=0.5, fy=0.5))
        
    return warped_card

def extract_face_roi(warped_card_image, roi_factors):
    """Extracts the face ROI from the warped card image."""
    if warped_card_image is None:
        return None
    h, w = warped_card_image.shape[:2]
    ymin_f, xmin_f, ymax_f, xmax_f = roi_factors
    
    ymin = int(h * ymin_f)
    xmin = int(w * xmin_f)
    ymax = int(h * ymax_f)
    xmax = int(w * xmax_f)
    
    if not (0 <= ymin < ymax <= h and 0 <= xmin < xmax <= w):
        print(f"Warning: Calculated Face ROI is out of bounds or invalid: ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax} for card size {h}x{w}")
        return None
        
    face_image = warped_card_image[ymin:ymax, xmin:xmax]
    return face_image