# card_preprocessor.py
import cv2
import numpy as np
from utils import perspective_transform, display_image # Assuming utils.py is in the same directory or package

def find_card_contour(image, debug=False):
    """Finds the largest rectangular contour, assumed to be the ID card."""
    orig_height, orig_width = image.shape[:2]

    # Resize for faster processing, keeping aspect ratio
    processing_height = 600.0
    ratio = orig_height / processing_height
    img_resized = cv2.resize(image, (int(orig_width / ratio), int(processing_height)))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    if debug: display_image("1. Resized Gray", gray)

    # --- MODIFICATION 1: Stronger, different blurring ---
    # Option A: Stronger GaussianBlur
    # blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # Option B: MedianBlur - often good for this kind of texture/noise
    blurred = cv2.medianBlur(gray, 7) # ksize must be odd and >1. 5, 7, 9 are common.
    # blurred = cv2.bilateralFilter(gray, 9, 75, 75) # Keep your original or try others
    if debug: display_image("2. Resized Blurred", blurred)

    # --- MODIFICATION 2: Potentially adjust Canny thresholds based on blurring ---
    # If median blur is strong, these might be okay or even need to be slightly higher
    # If the card edge is very faint, you might need lower.
    # Let's try with your current modified ones first, or slightly higher low threshold.
    edged = cv2.Canny(blurred, 50, 150) # MODIFIED: Canny thresholds (experiment here)
    # edged = cv2.Canny(blurred, 40, 120) # Your previous
    if debug: display_image("3. Resized Edged", edged)

    # Morphological operations to close gaps
    # --- MODIFICATION 3: Kernel for closing ---
    # Your (9,9) and 3 iterations is already strong. Let's keep it for now.
    # If edges are too thick and merging, you might reduce kernel or iterations.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Alternative: A bit less aggressive closing if (9,9)x3 is too much
    # kernel_alt = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    # closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_alt, iterations=2)

    if debug: display_image("4. Closed Edges", closed)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug: print("No contours found after closing.")
        return None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # Get top 10 largest

    card_contour_poly = None
    min_contour_area_ratio = 0.05 # MODIFIED: Slightly reduce min area requirement (5% of image)
                                  # The card is dominant, so 0.1 was probably fine, but more tolerance might help.
    min_contour_area = (img_resized.shape[0] * img_resized.shape[1]) * min_contour_area_ratio

    if debug:
        print(f"Min required contour area: {min_contour_area}")
        img_resized_contours_debug = img_resized.copy()
        cv2.drawContours(img_resized_contours_debug, contours, -1, (0,0,255), 1) # Draw all top 10
        display_image("5. Top 10 Contours", img_resized_contours_debug)


    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if debug: print(f"Contour {i} area: {area}")
        if area < min_contour_area:
            if debug: print(f"Contour {i} area {area} is less than min_area {min_contour_area}. Skipping.")
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # Epsilon can be tuned (0.01 to 0.05)

        if debug:
            print(f"Contour {i} has {len(approx)} points after approxPolyDP")
            temp_img = img_resized.copy()
            cv2.drawContours(temp_img, [approx], -1, (255,0,0), 2)
            display_image(f"6. Approx Poly for Contour {i}", temp_img)


        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            if debug: print(f"Found 4-sided contour {i}. Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")

            # Standard ID card aspect ratio is around 1.586 (e.g., 85.6mm / 53.98mm)
            # Allow some tolerance for perspective distortion.
            if 1.2 < aspect_ratio < 2.0 or 1/2.0 < aspect_ratio < 1/1.2: # Wider tolerance
                card_contour_poly = approx
                if debug:
                    print(f"Found potential card contour (Contour {i}) with area {area} and aspect ratio {aspect_ratio:.2f}")
                break
            elif debug:
                print(f"Contour {i} rejected due to aspect ratio: {aspect_ratio:.2f}")
        elif debug:
            print(f"Contour {i} rejected, not 4 points: {len(approx)} points")


    if card_contour_poly is None:
        if debug:
            print("No suitable 4-sided contour found meeting criteria.")
            # The "All Top Contours (Debug)" is already shown if contours were found.
            # If no contours at all, this part isn't reached for that image.
        return None

    if debug:
        debug_img_contour = img_resized.copy()
        cv2.drawContours(debug_img_contour, [card_contour_poly], -1, (0, 255, 0), 2)
        display_image("7. Final Card Contour on Resized", debug_img_contour) # Changed window name

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