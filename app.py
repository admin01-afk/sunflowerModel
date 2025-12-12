import math
import shapely.geometry as geom
import shapely.affinity as aff
import cv2
import numpy as np
from ultralytics import YOLO
import sys

"""
ToDO:
    Greenness metric to chlorophyll mapping
    health estimation
    leaf area estimation
    sunflower head area estimation
"""

# Chroma-key threshold: above this is considered "vegetation"
CHROMA_THRESHOLD = 0.6

def compute_yellowness(bgr):
    """
    bgr: (N,3) or (H,W,3) uint8/float array
    returns: per-pixel yellowness (same ordering as input flattened to (N,3))
    """
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 3)
    B = arr[:, 0]
    G = arr[:, 1]
    R = arr[:, 2]
    return (R + G) / 2.0 - B # simple yellowness index

def compute_greenness(bgr):
    """
    bgr: (N,3) or (H,W,3)
    returns: G / (R + B) ratio for each pixel (clipped to avoid outliers)
    """
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 3)

    B = arr[:, 0]
    G = arr[:, 1]
    R = arr[:, 2]

    denom = R + B
    denom = np.where(denom < 1e-3, 1e-3, denom) # clamp small denominators to avoid extreme ratios

    ratio = G / denom
    ratio = np.clip(ratio, 0.0, 50.0) # clip extreme outliers for stability (tweak upper bound if needed)
    return ratio

def chroma_key_green(bgr_image, threshold=CHROMA_THRESHOLD):
    """
    Produce HxW binary mask (0/1) where pixels are considered green.
    Uses the same compute_greenness routine.
    """
    flat = bgr_image.reshape(-1, 3).astype(np.float32)
    green_ratio = compute_greenness(flat)
    mask = (green_ratio > threshold)
    return mask.reshape(bgr_image.shape[:2]).astype(np.uint8)

def compute_chlorophyll_estimate(greenness_score):
    """
    function to convert greenness score to chlorophyll estimate.
    Leaves with high chlorophyll look almost the same as medium-chlorophyll leaves
    so logarithmic mapping
    """
    return 40 * np.log(greenness_score + 1) #Logarithmic mapping

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 app.py image.jpg")
        return

    image_path = sys.argv[1]
    model = YOLO("best.pt")
    img = cv2.imread(image_path)

    if img is None:
        print("Cannot load:", image_path)
        return

    h, w = img.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)  # will contain union of predicted polygons
    draw_img = img.copy()

    results = model(img)
    inside_yellowness_scores = []

    # ---------------------------------
    # Process segmentation polygons
    # ---------------------------------
    for result in results:
        if result.masks is None:
            continue

        for mask in result.masks:
            poly = mask.xy[0].astype(np.int32) # polygon points as Nx2 int array

            # polygon mask
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [poly], 255)

            # update global mask (union)
            full_mask = cv2.bitwise_or(full_mask, obj_mask)

            # compute yellowness for pixels inside this polygon
            inside_raw = img[obj_mask == 255]
            if inside_raw.size == 0:
                yscore = 0.0
            else:
                inside_pixels = inside_raw.reshape(-1, 3)
                yellow_vals = compute_yellowness(inside_pixels)
                yscore = float(np.mean(yellow_vals))

            inside_yellowness_scores.append(yscore)

            # draw filled blue overlay for object (YOLO-style)
            color_mask = np.zeros_like(img)
            color_mask[obj_mask == 255] = (255, 0, 0)  # blue fill
            draw_img = cv2.addWeighted(draw_img, 1.0, color_mask, 0.35, 0)

            # label at polygon centroid
            cx, cy = np.mean(poly, axis=0).astype(int)
            cv2.putText(draw_img, f"{yscore:.1f}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ---------------------------------
    # Compute greenness for pixels that are:
    #  - outside polygons (full_mask == 0)
    #  - AND pass the chroma-key green test
    # ---------------------------------
    # create chroma-key mask (H x W) using same threshold
    chroma_mask = chroma_key_green(img, threshold=CHROMA_THRESHOLD)

    # restrict to outside polygons
    chroma_mask[full_mask == 255] = 0

    # extract outside+green pixels
    outside_green_raw = img[chroma_mask == 1]
    if outside_green_raw.size == 0:
        outside_greenness = 0.0
        outside_greenness_median = 0.0
        outside_greenness_mean = 0.0
    else:
        outside_green_pixels = outside_green_raw.reshape(-1, 3).astype(np.float32)
        green_vals = compute_greenness(outside_green_pixels)  # (N,)
        outside_greenness_mean = float(np.mean(green_vals)) if green_vals.size > 0 else 0.0
        outside_greenness_median = float(np.median(green_vals)) if green_vals.size > 0 else 0.0
        outside_greenness = outside_greenness_mean

    # ---------------------------------
    # Visual overlay: faint green on top of draw_img
    # ---------------------------------
    green_overlay = np.zeros_like(img)
    green_overlay[chroma_mask == 1] = (0, 255, 0)  # solid green overlay
    blended = cv2.addWeighted(draw_img, 1.0, green_overlay, 0.35, 0)

    # annotation text
    h, w = blended.shape[:2]
    scale = max(0.6, w / 600)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blended, f"Greenness(avg)={outside_greenness:.2f}", (10, int(30*scale)),
            font, 0.7*scale, (0,0,0), int(2*scale))

    cv2.putText(blended, f"Greenness(med)={outside_greenness_median:.2f}", (10, int(60*scale)),
                font, 0.6*scale, (0,0,0), int(1*scale))

    # ---------------------------------
    # Console output
    # ---------------------------------
    print("\n=============================")
    print("      ANALYSIS RESULTS")
    print("=============================\n")

    for i, yscore in enumerate(inside_yellowness_scores):
        print(f"Polygon {i} Yellowness Score: {yscore:.2f}")

    print(f"\nOutside-Polygon Greenness (mean): {outside_greenness:.4f}")
    print(f"Outside-Polygon Greenness (median): {outside_greenness_median:.4f}")
    print(f"Outside-Polygon Greenness (mean, debug): {outside_greenness_mean:.4f}\n")
    print(f"Estimated Chlorophyll: {compute_chlorophyll_estimate(outside_greenness):.2f}\n")

    combined = np.hstack((img, blended))
    # Save output
    save_path = "prediction_"+ image_path.split("/")[-1]
    cv2.imwrite(save_path, combined)
    print(f"Saved result to: {save_path}")
    cv2.destroyAllWindows()

    # ---------------------------------
    # Show original + result side-by-side
    # ---------------------------------
    cv2.imshow("Analysis", combined)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()