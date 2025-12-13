import math
import shapely.geometry as geom
import cv2
import numpy as np
from ultralytics import YOLO
import argparse

"""
ToDO:
    sunflower head area estimation
"""

# Chroma-key threshold: above this is considered "vegetation"
CHROMA_THRESHOLD = 0.6

# returns yellowness index for each pixel in BGR array
def compute_yellowness(bgr):
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 3)
    B = arr[:, 0]
    G = arr[:, 1]
    R = arr[:, 2]
    return (R + G) / 2.0 - B

# returns greenness index for each pixel in BGR array
def compute_greenness(bgr):
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 3)
    B = arr[:, 0]
    G = arr[:, 1]
    R = arr[:, 2]
    denom = R + B
    denom = np.where(denom < 1e-3, 1e-3, denom)
    ratio = G / denom
    ratio = np.clip(ratio, 0.0, 50.0)
    return ratio

# chroma keying to extract green vegetation mask
def chroma_key_green(bgr_image, threshold=CHROMA_THRESHOLD):
    flat = bgr_image.reshape(-1, 3).astype(np.float32)
    green_ratio = compute_greenness(flat)
    mask = (green_ratio > threshold)
    return mask.reshape(bgr_image.shape[:2]).astype(np.uint8)

# logarithmic estimation of chlorophyll from greenness score
def compute_chlorophyll_estimate(greenness_score):
    return 40 * np.log(greenness_score + 1)

# compute ground intersection points of camera FOV, ideally 4 points
def camera_ground_intersection(H, tilt_deg, fov_v_deg, fov_h_deg):
    tilt  = math.radians(tilt_deg) #Python trig functions use radians, not degrees.
    fov_v = math.radians(fov_v_deg)
    fov_h = math.radians(fov_h_deg)

    # Tilt is downward, so pitch positive means down
    pitch_center = tilt

    # FOV offsets (vertical dv positive = downward)
    corners = [
        (-fov_v/2, -fov_h/2),
        (-fov_v/2, +fov_h/2),
        (+fov_v/2, -fov_h/2),
        (+fov_v/2, +fov_h/2),
    ]

    pts = []
    for dv, dh in corners:
        pitch = pitch_center + dv
        yaw = dh

        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = -math.sin(pitch)  # negative = downward

        if dz >= 0:
            continue

        t = -H / dz
        xg = dx * t
        yg = dy * t
        pts.append((xg, yg))

    return pts

# compute land area in viewport of the camera from camera&land parameters
def visible_land_area(width=1.4, length=7.0,#land dimensions in meters
                      camera_height=2.0,
                      camera_distance=1.4,
                      tilt_deg=15,
                      fov_v_deg=60,
                      fov_h_deg=80):
    
    """ 
    Diagram (not to scale):

            ↑ x
       ┌────|────┐
       │    |    │
       │  LAND   │   (length along y)
       │    |    │
       └────|────┘
            │
            │ camera_distance(1.4m)
            │
    up:z+   ● --------------------→ y
            Camera position(0,0,H)
    """
    # Important: Camera looks from the width side, mixing width/length would return wrong area.

    footprint = camera_ground_intersection(camera_height, tilt_deg, fov_v_deg, fov_h_deg)

    if len(footprint) < 3:
        return 0.0

    # sort footprint to form a valid polygon
    cx = sum(p[0] for p in footprint) / len(footprint)
    cy = sum(p[1] for p in footprint) / len(footprint)
    footprint_sorted = sorted(footprint, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    cam_poly = geom.Polygon(footprint_sorted)

    # Land polygon (ground coords)
    land_poly = geom.Polygon([
        (camera_distance, -width/2),
        (camera_distance + length, -width/2),
        (camera_distance + length, +width/2),
        (camera_distance, +width/2),
    ])

    intersect = cam_poly.intersection(land_poly)
    return intersect.area

def estimate_leaf_area(chroma_mask, camera_area_m2):
    green_pixels = np.sum(chroma_mask == 1)
    total_pixels = chroma_mask.size

    if total_pixels == 0 or camera_area_m2 <= 0:
        return 0.0

    pixel_fraction = green_pixels / total_pixels
    estimated_leaf_area = camera_area_m2 * pixel_fraction
    return estimated_leaf_area

def estimate_health_index(leaf_area, visible_land_area, chlorophyll_estimate, yellowness_score):
    """
    Normalize inputs before combining.
    chlorophyll_estimate expected roughly 0..~40, leaf_density 0..1, yellowness_score arbitrary.
    This function normalizes and returns 0..100.
    """
    if visible_land_area <= 0:
        return 0.0

    leaf_density = min(max(leaf_area / visible_land_area, 0.0), 1.0)

    # normalize chlorophyll (assume 0..40 map to 0..1)
    chl_norm = np.clip(chlorophyll_estimate / 40.0, 0.0, 1.0)

    # normalize yellowness: map expected range to 0..1 (tweak if needed)
    y = float(yellowness_score)
    y_norm = 1.0 - np.clip((y - 20.0) / 80.0, 0.0, 1.0)  # higher yellowness -> worse

    health = (0.45 * chl_norm + 0.35 * leaf_density + 0.20 * y_norm) * 100.0
    return float(np.clip(health, 0.0, 100.0))

def estimate_head_area_fraction(polygons,
                                chroma_mask,
                                image_shape):
    H, W = image_shape

    land_mask = (chroma_mask == 1)
    land_pixels = np.count_nonzero(land_mask)
    if land_pixels == 0:
        return 0.0

    head_mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        cv2.fillPoly(head_mask, [pts], 1)

    head_on_land = np.count_nonzero((head_mask == 1) & land_mask)
    return float(np.clip(head_on_land / land_pixels, 0.0, 1.0))

def parse_args():
    p = argparse.ArgumentParser(description="Sunflower analysis: head/leaf/health from an image")
    p.add_argument("image", help="path to image.jpg")
    p.add_argument("-g", "--gimbal-degree", type=float, default=15.0,
                   help="gimbal downward tilt in degrees (default: 15)")
    return p.parse_args()

def main():
    args = parse_args()
    image_path = args.image
    gimbal_deg = float(args.gimbal_degree)

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

    blue_overlay = np.zeros_like(img)

    # Process segmentation polygons
    for result in results:
        if result.masks is None:
            continue
        for mask in result.masks:
            poly = mask.xy[0].astype(np.int32)

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [poly], 255)
            full_mask = cv2.bitwise_or(full_mask, obj_mask)
            blue_overlay[obj_mask == 255] = (255, 0, 0)

            inside_raw = img[obj_mask == 255]
            if inside_raw.size == 0:
                yscore = 0.0
            else:
                inside_pixels = inside_raw.reshape(-1, 3)
                yellow_vals = compute_yellowness(inside_pixels)
                yscore = float(np.mean(yellow_vals))

            inside_yellowness_scores.append(yscore)

            color_mask = np.zeros_like(img)
            color_mask[obj_mask == 255] = (255, 0, 0)
            draw_img = cv2.addWeighted(draw_img, 1.0, color_mask, 0.35, 0)

            cx, cy = np.mean(poly, axis=0).astype(int)
            # draw black outline
            cv2.putText(draw_img, f"{yscore:.1f}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(draw_img, f"{yscore:.1f}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # chroma masks
    chroma_mask_all = chroma_key_green(img, threshold=CHROMA_THRESHOLD)
    chroma_mask_outside = chroma_mask_all.copy()
    chroma_mask_outside[full_mask == 255] = 0

    # outside greenness calculation
    outside_green_raw = img[chroma_mask_outside == 1]
    if outside_green_raw.size == 0:
        outside_greenness = 0.0
        outside_greenness_median = 0.0
        outside_greenness_mean = 0.0
    else:
        outside_green_pixels = outside_green_raw.reshape(-1, 3).astype(np.float32)
        green_vals = compute_greenness(outside_green_pixels)
        outside_greenness_mean = float(np.mean(green_vals)) if green_vals.size > 0 else 0.0
        outside_greenness_median = float(np.median(green_vals)) if green_vals.size > 0 else 0.0
        outside_greenness = outside_greenness_mean

    # visual overlay
    green_overlay = np.zeros_like(img)
    green_overlay[chroma_mask_all == 1] = (0, 255, 0)

    base = cv2.addWeighted(draw_img, 1.0, green_overlay, 0.25, 0)
    blended = cv2.addWeighted(base, 1.0, blue_overlay, 0.6, 0)

    # annotation text
    scale = max(0.6, w / 600)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blended, f"Greenness(avg)={outside_greenness:.2f}", (10, int(30*scale)),
                font, 0.7*scale, (0,0,0), int(2*scale))
    cv2.putText(blended, f"Greenness(med)={outside_greenness_median:.2f}", (10, int(60*scale)),
                font, 0.6*scale, (0,0,0), int(1*scale))

    # Console output & metrics
    analysis_result = ""
    analysis_result += "=============================\n"
    analysis_result += "      ANALYSIS RESULTS"
    analysis_result += "\n=============================\n"
    vis_area = visible_land_area(tilt_deg=gimbal_deg, fov_v_deg=60, fov_h_deg=80)
    leaf_area_est = estimate_leaf_area(chroma_mask_all, camera_area_m2=vis_area)

    # health index (normalized)
    health_index = estimate_health_index(
        leaf_area=leaf_area_est,
        visible_land_area=vis_area,
        chlorophyll_estimate=compute_chlorophyll_estimate(outside_greenness),
        yellowness_score=np.mean(inside_yellowness_scores) if inside_yellowness_scores else 0.0
    )

    # use gimbal_degree (from CLI) for visible area calculation
    analysis_result += f"THEORETICAL Visible Land Area: {vis_area:.2f} m^2, \ndepends ONLY on camera and land configuration\n"

    analysis_result += f"\nGreenness-Based Chlorophyll Index: {compute_chlorophyll_estimate(outside_greenness):.2f}\n"
    analysis_result += f"Estimated Health Index: {health_index:.2f}\n"

    analysis_result += f"Estimated Leaf Area: {leaf_area_est:.2f} m^2\n"
    # head fraction (use land mask that includes heads)
    land_mask_including_heads = (chroma_mask_all == 1) | (full_mask == 255)
    polygons_list = [mask.xy[0].astype(np.int32)
                     for result in results if result.masks is not None
                     for mask in result.masks]
    head_fraction = estimate_head_area_fraction(
        polygons=polygons_list,
        chroma_mask=land_mask_including_heads,
        image_shape=img.shape[:2]
    )
    analysis_result += f"Estimated Head Area Fraction: {head_fraction:.4f}"
    analysis_result += "\n=============================\n"

    print(analysis_result)

    combined = np.hstack((img, blended))
    save_path = "prediction_" + image_path.split("/")[-1]

    # save result image
    cv2.imwrite(save_path, combined)
    print(f"Saved result_image to: {save_path}")

    # save analysis result to a text file
    text_save_path = "analysis_" + image_path.split("/")[-1].rsplit(".", 1)[0] + ".txt"
    with open(text_save_path, "w") as f:
        f.write(analysis_result)
    print(f"Saved analysis result to: {text_save_path}")

    cv2.destroyAllWindows()

    cv2.imshow("Analysis", combined)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()