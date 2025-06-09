# card_preprocessor.py
import cv2
import numpy as np
from utils import perspective_transform, display_image # Assuming utils.py is in the same directory or package


def find_card_contour(image, debug=False):
    """Finds the largest rectangular contour, assumed to be the ID card."""
    orig_height, orig_width = image.shape[:2]

    processing_height = 600.0
    ratio = orig_height / processing_height
    img_resized = cv2.resize(image, (int(orig_width / ratio), int(processing_height)))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    if debug: display_image("1. Resized Gray", gray)

    # --- MODIFICATION 1: STRONGER BLURRING ---
    # blurred = cv2.medianBlur(gray, 7) # Your previous
    # Try a significantly stronger blur. Median blur is good.
    # Experiment with kernel size: 11, 15, 21. Must be odd.
    blurred = cv2.medianBlur(gray, 15) # Increased median blur kernel
    # Alternative: GaussianBlur with a large kernel
    # blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    if debug: display_image("2. Blurred (Stronger)", blurred)

    # --- MODIFICATION 2: THRESHOLDING to create a binary mask of the card ---
    # The goal is to make the card a white blob on a black background (or vice-versa).
    # The card appears lighter than many background elements and internal text.
    # Option A: Manual Thresholding (requires tuning `thresh_val`)
    # thresh_val = 130 # EXPERIMENT with this value (e.g., 100, 120, 140, 150, 160)
    # _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    # If card becomes black, use cv2.THRESH_BINARY_INV
    # _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Option B: Otsu's Binarization (often works well if there's a bimodal histogram)
    # We expect the card to be the lighter part.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Check if Otsu made the card black (if background is very light)
    # A simple check: if the mean of the threshold image is low, it might be inverted.
    # The card should be the larger area. If most of image is black, card is black.
    if np.mean(thresh) < 100: # Heuristic: if less than ~40% white pixels
         thresh = cv2.bitwise_not(thresh) # Invert if card became black

    # Option C: Adaptive Thresholding (can be good for uneven illumination)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                               cv2.THRESH_BINARY_INV, 21, 5) # blockSize=21 (odd), C=5 (constant)
                                  # Use THRESH_BINARY_INV if you want white objects on black background

    if debug: display_image("3. Thresholded", thresh)

    # --- MODIFICATION 3: Canny on the THRESHOLDED image ---
    # The edges should now be much cleaner, primarily the outline of the card.
    # Canny thresholds might need to be adjusted based on the clarity of 'thresh'.
    # Using 30, 75 as a starting point for potentially cleaner edges.
    edged = cv2.Canny(thresh, 30, 75) # MODIFIED Canny for thresholded input
    if debug: display_image("4. Edged (from Thresholded)", edged)

    # --- MODIFICATION 4: Morphological Closing ---
    # If 'edged' is cleaner, we might use slightly less aggressive closing,
    # or keep it strong to ensure connectivity.
    # Let's try a slightly smaller kernel or fewer iterations first.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # Reduced from (9,9)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_close, iterations=2) # Reduced iterations
    # If gaps persist, you can revert to (9,9) and 3 iterations, or even increase.
    if debug: display_image("5. Closed Edges", closed)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug: print("No contours found after closing.")
        return None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # Top 5 should be enough

    card_contour_poly = None
    # min_contour_area_ratio = 0.05 # Your previous
    min_contour_area_ratio = 0.10 # Card should be at least 10% of the image area
    min_contour_area = (img_resized.shape[0] * img_resized.shape[1]) * min_contour_area_ratio

    if debug:
        print(f"Min required contour area: {min_contour_area:.2f} (based on {min_contour_area_ratio*100}%)")
        img_resized_contours_debug = img_resized.copy()
        cv2.drawContours(img_resized_contours_debug, contours, -1, (0,0,255), 1)
        display_image("6. Top Contours from Closed", img_resized_contours_debug)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if debug: print(f"Contour {i} area: {area:.2f}")
        if area < min_contour_area:
            if debug: print(f"Contour {i} area {area:.2f} is less than min_area {min_contour_area:.2f}. Skipping.")
            continue

        peri = cv2.arcLength(c, True)
        # Epsilon for approxPolyDP. 0.02 is common.
        # If corners are too rounded by blur/threshold, may need larger epsilon (e.g., 0.03, 0.04)
        # If too many points (wavy edge), increase epsilon.
        epsilon = 0.02 * peri # Start with 0.02
        approx = cv2.approxPolyDP(c, epsilon, True)

        if debug:
            print(f"Contour {i} has {len(approx)} points after approxPolyDP (epsilon: {epsilon:.2f})")
            temp_img = img_resized.copy()
            cv2.drawContours(temp_img, [approx], -1, (255,0,0), 2) # Draw current approx in blue
            display_image(f"7. Approx Poly for Contour {i}", temp_img)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            if debug: print(f"Found 4-sided contour {i}. Area: {area:.2f}, Aspect Ratio: {aspect_ratio:.2f}")

            # Wider tolerance for aspect ratio
            if 1.2 < aspect_ratio < 2.0 or (h > w and 1/2.0 < aspect_ratio < 1/1.2) :
                card_contour_poly = approx
                if debug:
                    print(f"SUCCESS: Found potential card (Contour {i}) with area {area:.2f} and aspect ratio {aspect_ratio:.2f}")
                break
            elif debug:
                print(f"Contour {i} rejected due to aspect ratio: {aspect_ratio:.2f} (Expected ~1.58 or ~0.63)")
        elif debug:
            print(f"Contour {i} rejected, not 4 points: {len(approx)} points")


    if card_contour_poly is None:
        if debug:
            print("No suitable 4-sided contour found meeting criteria after all checks.")
        return None

    if debug:
        debug_img_contour = img_resized.copy()
        cv2.drawContours(debug_img_contour, [card_contour_poly], -1, (0, 255, 0), 2)
        display_image("8. Final Card Contour on Resized", debug_img_contour)

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