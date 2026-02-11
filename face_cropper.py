"""
Minimal OpenCV face detector + head cropper.

Only required inputs:
    1. Input directory containing images
    2. Output directory for cropped images

Defaults:
- Uses Haar cascade frontal face detector
- Crops largest detected face
- Expands crop to include full head (hair + slight neck)
- Preserves folder structure
"""

from pathlib import Path
import cv2


# ----------------------------
# Simple user input
# ----------------------------

input_dir = Path(input("Enter input directory path: ").strip().strip('"'))
output_dir = Path(input("Enter output directory path: ").strip().strip('"'))

if not input_dir.exists() or not input_dir.is_dir():
    raise ValueError("Input directory does not exist or is not a directory.")

output_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Load Haar cascade
# ----------------------------

cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(str(cascade_path))

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade.")


# ----------------------------
# Helper functions
# ----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clamp(val, low, high):
    return max(low, min(high, val))


def expand_to_head(x, y, w, h, img_w, img_h):
    """
    Expand detected face box to better include full head.
    Tuned constants for typical portrait images.
    """
    scale_w = 1.6
    scale_h = 2.6
    shift_up = 0.10

    cx = x + w / 2
    cy = y + h / 2 - (h * shift_up)

    new_w = w * scale_w
    new_h = h * scale_h

    x1 = int(cx - new_w / 2)
    y1 = int(cy - new_h / 2)
    x2 = int(cx + new_w / 2)
    y2 = int(cy + new_h / 2)

    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, 1, img_w)
    y2 = clamp(y2, 1, img_h)

    return x1, y1, x2, y2


# ----------------------------
# Process images
# ----------------------------

processed = 0
skipped = 0
skip_list = []

for img_path in input_dir.rglob("*"):
    total = len(list(input_dir.rglob("*")))
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        skipped += 1
        print("skipped" + str(img_path))
        skip_list.append(str(img_path))
        continue
    progress = processed + skipped
    print(f"Progress...[{progress} of {total}]")

    # Use largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = expand_to_head(x, y, w, h, img_w, img_h)

    crop = img[y1:y2, x1:x2]

    # Preserve folder structure
    relative_path = img_path.relative_to(input_dir)
    save_path = (output_dir / relative_path).with_name(
        relative_path.stem + "_crop" + relative_path.suffix
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if cv2.imwrite(str(save_path), crop):
        processed += 1
    else:
        skipped += 1

print(f"Done. Processed: {processed}, Skipped: {skipped}")
print(f"skipped files: {skip_list}")