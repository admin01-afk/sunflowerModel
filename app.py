import cv2
import numpy as np
from ultralytics import YOLO
import sys


def compute_yellowness(bgr):
    B, G, R = bgr[:, 0], bgr[:, 1], bgr[:, 2]
    return (R + G) / 2 - B


def compute_greenness(bgr):
    """
    bgr: array shape (N,3) or (H,W,3) depending on caller.
    Returns ratio = G / (R+B) as float32, with denom clamped and extreme values clipped.
    """
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        # convert H,W,3 -> N,3
        arr = arr.reshape(-1, 3)

    B = arr[:, 0]
    G = arr[:, 1]
    R = arr[:, 2]

    # avoid divide-by-zero and very small denominators
    denom = R + B
    denom = np.where(denom < 1e-3, 1e-3, denom)

    ratio = G / denom

    # Clip obvious outliers so mean/median are stable — tune upper_clip if you want
    ratio = np.clip(ratio, 0.0, 20.0)

    return ratio  # shape (N,)


def chroma_key_green(bgr_image, threshold=0.6):
    """
    bgr_image: HxWx3 uint8
    returns: mask HxW uint8 (0/1)
    """
    flat = bgr_image.reshape(-1, 3).astype(np.float32)
    green_ratio = compute_greenness(flat)
    mask = (green_ratio > threshold)
    return mask.reshape(bgr_image.shape[:2]).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py image.jpg")
        return

    image_path = sys.argv[1]
    model = YOLO("best.pt")
    img = cv2.imread(image_path)

    if img is None:
        print("Cannot load:", image_path)
        return

    h, w = img.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    draw_img = img.copy()

    results = model(img)
    inside_yellowness_scores = []

    # ------------------------------
    # Process segmentation
    # ------------------------------
    for result in results:
        if result.masks is None:
            continue

        for mask in result.masks:
            poly = mask.xy[0].astype(np.int32)

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [poly], 255)

            full_mask |= obj_mask

            # ----- Yellowness -----
            inside_pixels = img[obj_mask == 255].reshape(-1, 3)
            yellow_vals = compute_yellowness(inside_pixels)
            yscore = float(np.mean(yellow_vals))
            inside_yellowness_scores.append(yscore)

            # ----- Blue filled mask -----
            color_mask = np.zeros_like(img)
            color_mask[obj_mask == 255] = (255, 0, 0)

            draw_img = cv2.addWeighted(draw_img, 1.0, color_mask, 0.35, 0)

            # label
            x, y = np.mean(poly, axis=0).astype(int)
            cv2.putText(draw_img, f"{yscore:.1f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    # ------------------------------
    # Greenness outside polygons
    # ------------------------------
    # Extract outside region safely
    outside_raw = img[full_mask == 0]  # may be (N,3) or flattened depending on numpy version
    if outside_raw.size == 0:
        outside_pixels = np.zeros((0, 3), dtype=np.float32)
    else:
        # Ensure shape (N,3)
        outside_pixels = outside_raw.reshape(-1, 3).astype(np.float32)

    outside_greenness = 0.0
    if len(outside_pixels) > 0:
        green_vals = compute_greenness(outside_pixels)  # shape (N,)
        # use the same threshold as chroma-key
        mask_green = green_vals > 1.8

        # debug info (optional) - uncomment if you want console diagnostics
        # print("DEBUG outside_pixels.shape:", outside_pixels.shape)
        # print("DEBUG green_vals: min", np.min(green_vals), "p50", np.median(green_vals), "p99", np.percentile(green_vals, 99))

        if np.any(mask_green):
            # Use median to avoid outlier influence, also compute mean for reference
            outside_greenness = float(np.median(green_vals[mask_green]))
            outside_greenness_mean = float(np.mean(green_vals[mask_green]))
        else:
            outside_greenness = 0.0
        print("Outside greenness (median, mean):", outside_greenness, outside_greenness_mean)

    # ------------------------------
    # Green faint overlay
    # ------------------------------
    green_mask = chroma_key_green(img)
    green_mask[full_mask == 255] = 0

    green_overlay = np.zeros_like(img)
    green_overlay[green_mask == 1] = (0, 255, 0)

    blended = cv2.addWeighted(draw_img, 1.0, green_overlay, 0.35, 0)

    cv2.putText(blended, f"Greenness={outside_greenness:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    # ------------------------------
    # Show original + result
    # ------------------------------
    combined = np.hstack((img, blended))
    cv2.imshow("Analysis", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------------------
    # Console print
    # ------------------------------
    print("\n=============================")
    print("      ANALYSIS RESULTS")
    print("=============================\n")

    for i, yscore in enumerate(inside_yellowness_scores):
        print(f"Polygon {i} Yellowness Score: {yscore:.2f}")

    print(f"\nOutside-Polygon Greenness Score: {outside_greenness:.2f}\n")


if __name__ == "__main__":
    main()