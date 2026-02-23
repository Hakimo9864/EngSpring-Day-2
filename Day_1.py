"""
AI with Embedded Vision — Day 1 Starter
Topic: Images as data + thresholding as a classifier (laptop-only)

Requirements:
  pip install opencv-python numpy

Usage examples:
  python day1_threshold_lab.py --image sample.jpg
  python day1_threshold_lab.py --image sample.jpg --thresholds 60 120 180
  python day1_threshold_lab.py --image sample.jpg --no-gui

Outputs:
  Creates an "outputs/" folder and saves:
    - grayscale.png
    - thresh_<T>.png for each threshold T
"""


import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to an image file (jpg/png).")
    p.add_argument(
        "--thresholds",
        type=int,
        nargs="*",
        default=[60, 120, 180],
        help="One or more integer thresholds in [0, 255]. Example: --thresholds 50 100 150",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to save output images.",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open display windows (useful on systems without GUI).",
    )
    return p.parse_args()


def ensure_thresholds(thresholds: List[int]) -> List[int]:
    cleaned: List[int] = []
    for t in thresholds:
        if t < 0 or t > 255:
            raise ValueError(f"Threshold {t} is out of range [0, 255].")
        cleaned.append(int(t))
    # Remove duplicates, preserve order
    seen = set()
    unique = []
    for t in cleaned:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def percent_white(binary_img: np.ndarray) -> float:
    """
    binary_img is expected to contain values 0 or 255.
    Returns percent of pixels that are white (255).
    """
    if binary_img.ndim != 2:
        raise ValueError("percent_white expects a single-channel (grayscale/binary) image.")
    white = np.count_nonzero(binary_img == 255)
    total = binary_img.size
    return 100.0 * white / total


def main() -> None:
    args = parse_args()
    thresholds = ensure_thresholds(args.thresholds)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Could not find image: {img_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load image (OpenCV loads as BGR by default)
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image (is it a valid jpg/png?): {img_path}")

    print("\n=== Image inspection ===")
    print(f"File: {img_path.name}")
    print(f"Shape: {image_bgr.shape}  (height, width, channels)")
    print(f"Data type: {image_bgr.dtype}  (usually uint8)")
    h, w, c = image_bgr.shape

    # Sample a pixel (row 100, col 200) if in bounds
    r, col = min(100, h - 1), min(200, w - 1)
    pixel_bgr = image_bgr[r, col]
    print(f"Pixel at (row={r}, col={col}) in BGR: {pixel_bgr.tolist()}")

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_path = outdir / "grayscale.png"
    cv2.imwrite(str(gray_path), gray)
    print(f"Saved grayscale image to: {gray_path}")

    # Threshold experiments
    print("\n=== Threshold experiments ===")
    for t in thresholds:
        # Binary threshold: pixels > t become 255, else 0
        _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        pw = percent_white(thresh)

        out_path = outdir / f"thresh_{t}.png"
        cv2.imwrite(str(out_path), thresh)

        print(f"Threshold {t:3d}: percent white = {pw:6.2f}%   saved -> {out_path.name}")

    # Optional display windows (press any key to advance, 'q' to quit early)
    if not args.no_gui:
        cv2.imshow("Original (BGR)", image_bgr)
        cv2.imshow("Grayscale", gray)
        print("\nGUI mode: close windows or press keys in an image window.")
        print("Tip: If windows don't appear on your system, rerun with --no-gui.\n")

        for t in thresholds:
            _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            cv2.imshow(f"Threshold {t}", thresh)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(f"Threshold {t}")
            if key == ord("q"):
                break

        cv2.destroyAllWindows()

    # Reflection prompts printed at end for convenience
    print("\n=== Reflection prompts (answer in your write-up) ===")
    print("1) When is a threshold classifier reasonable? When does it fail?")
    print("2) What assumptions does thresholding make about the image?")
    print("3) If you wanted to improve this classifier, what extra information (features) could you use?")
    print("4) Is thresholding machine learning? Why or why not?")


if __name__ == "__main__":
    main()