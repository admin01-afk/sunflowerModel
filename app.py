import math
import shapely.geometry as geom
import cv2
import numpy as np
from ultralytics import YOLO
import argparse

"""
Sunflower analysis: head/leaf/health from an image
- uses YOLOv8 segmentation masks (best.pt)
- produces annotated image and a `.txt` analysis summary
"""

# ----- Camera FOV constants (modify here if needed) -----
CAM_FOV_H = 80.0   # horizontal FOV in degrees
CAM_FOV_V = 60.0   # vertical FOV in degrees

# Chroma-key threshold: above this is considered "vegetation"
CHROMA_THRESHOLD = 0.6

# returns yellowness index for each pixel in BGR array
def compute_yellowness(bgr):
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 3)
    B = arr[:, 0]; G = arr[:, 1]; R = arr[:, 2]
    return (R + G) / 2.0 - B

# returns greenness index for each pixel in BGR array
def compute_greenness(bgr):
    arr = bgr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 3)
    B = arr[:, 0]; G = arr[:, 1]; R = arr[:, 2]
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

# logarithmic mapping from greenness to a relative chlorophyll index (no physical unit)
def compute_chlorophyll_estimate(greenness_score):
    return 40 * np.log(greenness_score + 1)

# compute ground intersection points of camera FOV
def camera_ground_intersection(H, tilt_deg, fov_v_deg, fov_h_deg):
    tilt  = math.radians(tilt_deg)
    fov_v = math.radians(fov_v_deg)
    fov_h = math.radians(fov_h_deg)

    pitch_center = tilt
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

# compute visible land area (intersection of camera footprint and land rectangle)
def visible_land_area(width=1.4, length=7.0,           # land dimensions in meters
                      camera_height=2.0,
                      camera_distance=1.4,
                      tilt_deg=15,
                      fov_v_deg=CAM_FOV_V,
                      fov_h_deg=CAM_FOV_H):
    footprint = camera_ground_intersection(camera_height, tilt_deg, fov_v_deg, fov_h_deg)
    if len(footprint) < 3:
        print("Critical: camera over horizon, less than 4 ground intersections, area calculation invalid.")
        return 0.0

    cx = sum(p[0] for p in footprint) / len(footprint)
    cy = sum(p[1] for p in footprint) / len(footprint)
    footprint_sorted = sorted(footprint, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    cam_poly = geom.Polygon(footprint_sorted)

    land_poly = geom.Polygon([
        (camera_distance, -width/2),
        (camera_distance + length, -width/2),
        (camera_distance + length, +width/2),
        (camera_distance, +width/2),
    ])

    intersect = cam_poly.intersection(land_poly)
    return intersect.area

# estimate leaf area from chroma-mask and camera-visible area (minimum estimate)
def estimate_leaf_area(chroma_mask, camera_area_m2):
    green_pixels = np.sum(chroma_mask == 1)
    total_pixels = chroma_mask.size
    if total_pixels == 0 or camera_area_m2 <= 0:
        return 0.0
    pixel_fraction = green_pixels / total_pixels
    estimated_leaf_area = camera_area_m2 * pixel_fraction
    return estimated_leaf_area

# combine normalized signals into a 0..100 health index
def estimate_health_index(leaf_area,
                          visible_land_area_m2,
                          chlorophyll_estimate,
                          yellowness_score):
    # ----- normalize leaf density -----
    if visible_land_area_m2 > 0:
        leaf_density = leaf_area / visible_land_area_m2
        leaf_density = np.clip(leaf_density, 0.05, 1.0)   # floor at 5%
        leaf_valid = True
    else:
        leaf_density = 0.5  # neutral fallback
        leaf_valid = False

    # ----- normalize chlorophyll index -----
    chl_norm = chlorophyll_estimate / 40.0
    chl_norm = np.clip(chl_norm, 0.05, 1.0)               # floor at 5%
    chl_valid = chlorophyll_estimate > 0

    # ----- normalize yellowness (higher = worse) -----
    y = float(yellowness_score)
    y_norm = 1.0 - np.clip((y - 20.0) / 80.0, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.05, 1.0)                   # floor at 5%
    y_valid = True

    # ----- weights -----
    weights = {
        "chl": 0.45,
        "leaf": 0.35,
        "yellow": 0.20
    }

    # disable invalid signals
    if not leaf_valid:
        weights["leaf"] = 0.0
    if not chl_valid:
        weights["chl"] = 0.0

    # renormalize weights
    w_sum = sum(weights.values())
    if w_sum <= 0:
        return 50.0  # fully unknown → neutral health

    for k in weights:
        weights[k] /= w_sum

    # ----- final score -----
    health = (
        weights["chl"]    * chl_norm +
        weights["leaf"]   * leaf_density +
        weights["yellow"] * y_norm
    ) * 100.0

    return float(np.clip(health, 0.0, 100.0))

# fraction of vegetation pixels that are covered by head polygons (0..1)
def estimate_head_area_fraction(polygons, chroma_mask, image_shape):
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
    p.add_argument("--camera-height", type=float, default=2.0,
                   help="camera height above ground in meters (default: 2.0)")
    p.add_argument("--camera-distance", type=float, default=1.4,
                   help="horizontal distance from camera to start of the land area in meters (default: 1.4)")
    p.add_argument("--land-width", type=float, default=1.4,
                   help="land width (short side) in meters (default: 1.4)")
    p.add_argument("--land-length", type=float, default=7.0,
                   help="land length (long side) in meters (default: 7.0)")
    return p.parse_args()

def main():
    args = parse_args()
    image_path = args.image
    gimbal_deg = float(args.gimbal_degree)
    cam_H = float(args.camera_height)
    cam_dist = float(args.camera_distance)
    land_w = float(args.land_width)
    land_l = float(args.land_length)

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

    # accumulate blue_overlay and polygons list
    blue_overlay = np.zeros_like(img)
    polygons_list = []
    labels = []

    # Process segmentation polygons
    for result in results:
        if result.masks is None:
            continue
        for mask in result.masks:
            poly = mask.xy[0].astype(np.int32)
            polygons_list.append(poly)

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

            # draw a blue-ish fill on draw_img for visual (will be re-applied above)
            color_mask = np.zeros_like(img)
            color_mask[obj_mask == 255] = (255, 0, 0)
            draw_img = cv2.addWeighted(draw_img, 1.0, color_mask, 0.35, 0)

            # store label to draw later
            cx, cy = np.mean(poly, axis=0).astype(int)
            labels.append((int(cx), int(cy), f"{yscore:.1f}"))

    # chroma masks (vegetation)
    chroma_mask_all = chroma_key_green(img, threshold=CHROMA_THRESHOLD)
    chroma_mask_outside = chroma_mask_all.copy()
    chroma_mask_outside[full_mask == 255] = 0

    # compute greenness-based chlorophyll index from outside-vegetation (outside heads)
    outside_green_raw = img[chroma_mask_outside == 1]
    if outside_green_raw.size == 0:
        outside_greenness = 0.0
    else:
        outside_green_pixels = outside_green_raw.reshape(-1, 3).astype(np.float32)
        green_vals = compute_greenness(outside_green_pixels)
        outside_greenness = float(np.mean(green_vals)) if green_vals.size > 0 else 0.0

    chlorophyll_index = compute_chlorophyll_estimate(outside_greenness)

    # visible land area (use constants CAM_FOV_H/CAM_FOV_V)
    vis_area = visible_land_area(width=land_w, length=land_l,
                                 camera_height=cam_H,
                                 camera_distance=cam_dist,
                                 tilt_deg=gimbal_deg,
                                 fov_v_deg=CAM_FOV_V, fov_h_deg=CAM_FOV_H)

    # estimated leaf area (minimum) using full chroma mask of the frame
    leaf_area_est = estimate_leaf_area(chroma_mask_all, camera_area_m2=vis_area)

    # health index (normalized 0..100)
    yellowness_mean = np.mean(inside_yellowness_scores) if inside_yellowness_scores else 0.0
    health_index = estimate_health_index(
        leaf_area=leaf_area_est,
        visible_land_area_m2=vis_area,
        chlorophyll_estimate=chlorophyll_index,
        yellowness_score=yellowness_mean
    )

    # head area fraction (fraction of vegetation pixels covered by heads)
    land_mask_including_heads = (chroma_mask_all == 1) | (full_mask == 255)
    head_fraction = estimate_head_area_fraction(polygons_list, land_mask_including_heads, img.shape[:2])

    # Visual overlay: green background then blue heads on top
    green_overlay = np.zeros_like(img)
    green_overlay[chroma_mask_all == 1] = (0, 255, 0)
    base = cv2.addWeighted(draw_img, 1.0, green_overlay, 0.25, 0)
    blended = cv2.addWeighted(base, 1.0, blue_overlay, 0.6, 0)

    # draw labels on final blended image (outline + fill)
    scale = max(0.8, w / 500)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, text) in labels:
        cv2.putText(blended, text, (x, y - int(10*scale)), font, 0.6*scale, (0,0,0), int(3*scale), cv2.LINE_AA)
        cv2.putText(blended, text, (x, y - int(10*scale)), font, 0.6*scale, (0,255,255), int(1*scale), cv2.LINE_AA)

    # annotation texts (the four requested labels)
    scale = max(0.6, w / 600)
    y0 = int(30 * scale)
    dy = int(28 * scale)
    cv2.putText(blended, f"Chl Index: {chlorophyll_index:.2f}", (10, y0),
                font, 0.6*scale, (0,0,0), int(2*scale), cv2.LINE_AA)
    cv2.putText(blended, f"Health Index: {health_index:.2f}", (10, y0 + dy),
                font, 0.6*scale, (0,0,0), int(2*scale), cv2.LINE_AA)
    cv2.putText(blended, f"Head%: {head_fraction:.4f}", (10, y0 + 2*dy),
                font, 0.6*scale, (0,0,0), int(2*scale), cv2.LINE_AA)
    if(leaf_area_est > 0):
        cv2.putText(blended, f"LeafArea: {leaf_area_est:.2f} m^2", (10, y0 + 3*dy),
                font, 0.6*scale, (0,0,0), int(2*scale), cv2.LINE_AA)

    # build textual analysis result and save
    analysis_result = []
    analysis_result.append("=============================")
    analysis_result.append("      ANALYSIS RESULTS")
    analysis_result.append("=============================")
    analysis_result.append((vis_area > 0) and (f"THEORETICAL Visible Land Area (m^2): {vis_area:.2f}\n") or "area calc invalid!\n")
    analysis_result.append(f"Greenness-Based Chlorophyll Index: {chlorophyll_index:.2f}")
    analysis_result.append(f"Estimated Health Index: {health_index:.2f}")
    analysis_result.append(f"Estimated Leaf Area (m^2): {leaf_area_est:.2f}")
    analysis_result.append(f"Estimated Head Area Fraction: {head_fraction:.4f}")
    analysis_text = "\n".join(analysis_result) + "\n"

    print(analysis_text)

    combined = np.hstack((img, blended))
    save_path = "prediction_" + image_path.split("/")[-1]
    cv2.imwrite(save_path, combined)
    print(f"Saved result_image to: {save_path}")

    text_save_path = "analysis_" + image_path.split("/")[-1].rsplit(".", 1)[0] + ".txt"
    with open(text_save_path, "w") as f:
        f.write(analysis_text)
    print(f"Saved analysis result to: {text_save_path}")

    cv2.destroyAllWindows()
    cv2.imshow("Analysis", combined)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()