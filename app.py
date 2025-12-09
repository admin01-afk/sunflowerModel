import sys
from ultralytics import YOLO
import cv2

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_segmentation.py <image.jpg>")
        return

    model_path = "best.pt"
    image_path = sys.argv[1]

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Run inference
    print(f"Running inference on: {image_path}")
    results = model(image_path)

    # YOLO returns a list; take first
    result = results[0]

    # Show/save plotted output
    output_img = result.plot()  # numpy uint8 BGR image
    cv2.imshow("YOLOv8 Segmentation Result", output_img)

    # Save output
    save_path = "prediction_"+ image_path
    cv2.imwrite(save_path, output_img)
    print(f"Saved result to: {save_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()