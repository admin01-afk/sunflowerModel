import cv2
import numpy as np
from ultralytics import YOLO
import sys


def compute_yellowness(bgr):
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return (R + G) / 2 - B


def compute_greenness(bgr):
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return G - np.maximum(R, B)


def chroma_key_green(bgr, threshold=0):
    """Returns a mask of green pixels."""
    green_strength = compute_greenness(bgr)
    return (green_strength > threshold).astype(np.uint8)


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

    # ---------------------------------
    # Process segmentation polygons
    # ---------------------------------
    for result in results:
        if result.masks is None:
            print("❗ No segmentation masks found!")
            continue

        for idx, mask in enumerate(result.masks):
            poly = mask.xy[0].astype(np.int32)

            # Mask for this polygon
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [poly], 255)

            # Add polygon to global mask
            full_mask = cv2.bitwise_or(full_mask, obj_mask)

            # Yellowness inside this polygon
            inside_pixels = img[obj_mask == 255]
            yellow_vals = compute_yellowness(inside_pixels.reshape(-1, 1, 3))
            yscore = float(np.mean(yellow_vals))
            inside_yellowness_scores.append(yscore)

            # Draw polygon on image
            cv2.polylines(draw_img, [poly], True, (0, 255, 255), 2)

            # Label
            x, y = poly[0]
            cv2.putText(draw_img, f"Y={yscore:.1f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---------------------------------
    # Compute greenness OUTSIDE polygons
    # ---------------------------------
    outside_pixels = img[full_mask == 0]
    if len(outside_pixels) > 0:
        green_vals_all = compute_greenness(outside_pixels.reshape(-1, 1, 3))
        green_pixels = green_vals_all.squeeze() > 0

        if np.any(green_pixels):
            outside_greenness = float(np.mean(green_vals_all[green_pixels]))
        else:
            outside_greenness = 0.0
    else:
        outside_greenness = 0.0

    # ---------------------------------
    # Make faint green overlay (chroma key style)
    # ---------------------------------
    green_mask = chroma_key_green(img)
    green_mask[full_mask == 255] = 0  # remove polygon regions

    green_overlay = np.zeros_like(img)
    green_overlay[green_mask == 1] = (0, 255, 0)  # solid green

    # Blend into main image
    blended = cv2.addWeighted(draw_img, 1.0, green_overlay, 0.35, 0)

    # Add greenness score text
    cv2.putText(blended, f"Greenness={outside_greenness:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    # ---------------------------------
    # Print console output
    # ---------------------------------
    print("\n=============================")
    print("      ANALYSIS RESULTS")
    print("=============================\n")

    for i, yscore in enumerate(inside_yellowness_scores):
        print(f"Polygon {i} Yellowness Score: {yscore:.2f}")

    print(f"\nOutside-Polygon Greenness Score: {outside_greenness:.2f}\n")

    # ---------------------------------
    # ONE IMAGE WINDOW ONLY
    # ---------------------------------
    combined = np.hstack((img, blended))
    cv2.imshow("Analysis", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()