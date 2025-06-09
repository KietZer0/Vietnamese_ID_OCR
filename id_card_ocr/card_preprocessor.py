# card_preprocessor.py
import cv2
import numpy as np
from utils import perspective_transform, display_image # Assuming utils.py is in the same directory or package

def find_card_contour(image, debug=False):
    """Finds the largest rectangular contour, assumed to be the ID card."""
    orig_height, orig_width = image.shape[:2]

    # Resize for faster processing, keeping aspect ratio
    # Target height for processing
    processing_height = 600.0
    ratio = orig_height / processing_height
    img_resized = cv2.resize(image, (int(orig_width / ratio), int(processing_height)))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Experiment with different blurring
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75) # Better at preserving edges

    # Experiment with Canny edge detection parameters
    # Lower thresholds might pick up more edges, higher thresholds fewer.
    # Adjusted from 75, 200. Let's try slightly lower thresholds for more sensitivity
    edged = cv2.Canny(blurred, 40, 120) # MODIFIED: Lowered Canny thresholds slightly

    if debug:
        display_image("Resized Gray", gray)
        display_image("Resized Blurred", blurred)
        display_image("Resized Edged", edged)

    # Morphological operations to close gaps and remove noise
    # MODIFIED: Increased kernel size and iterations for MORPH_CLOSE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)) # Increased from (5,5)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3) # Increased iterations from 2

    if debug:
        display_image("Closed Edges", closed)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    card_contour_poly = None
    # MODIFIED: Slightly reduce min_contour_area requirement, though 10% was likely fine.
    # The card in your image is large, so 0.1 should be okay. Let's keep it at 0.1.
    min_contour_area = (img_resized.shape[0] * img_resized.shape[1]) * 0.1

    for c in contours:
        if cv2.contourArea(c) < min_contour_area:
            continue

        peri = cv2.arcLength(c, True)
        # MODIFIED: Slightly adjust epsilon for approxPolyDP if needed, 0.02 is usually good.
        # Could try 0.018 or 0.022 if 4 points are not consistently found. Sticking to 0.02 for now.
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.3 < aspect_ratio < 1.9 or 1/1.9 < aspect_ratio < 1/1.3:
                card_contour_poly = approx
                if debug:
                    print(f"Found potential card contour with area {cv2.contourArea(c)} and aspect ratio {aspect_ratio}")
                break

    if card_contour_poly is None:
        if debug:
            print("No suitable 4-sided contour found.")
            debug_img_all_contours = img_resized.copy()
            cv2.drawContours(debug_img_all_contours, contours, -1, (0,0,255), 1)
            display_image("All Top Contours (Debug)", debug_img_all_contours) # Changed window name slightly for clarity
        return None

    if debug:
        debug_img_contour = img_resized.copy()
        cv2.drawContours(debug_img_contour, [card_contour_poly], -1, (0, 255, 0), 2)
        display_image("Card Contour on Resized (Debug)", debug_img_contour) # Changed window name

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
        display_image("Warped Card (Final Debug)", cv2.resize(warped_card, (0,0), fx=0.5, fy=0.5)) # Changed window name

    return warped_card

# extract_face_roi remains the same
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