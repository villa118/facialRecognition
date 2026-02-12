"""
Defaults:
- Uses Haar cascade frontal face detector
- Crops largest detected face
- Expands crop to include full head (hair + slight neck)
- Preserves folder structure
"""

from pathlib import Path
import cv2
import tkinter as tk
from tkinter.filedialog import askdirectory
import compare_pics
import time


# Get user input

ans = input("This script will ask you to do the following:\n"
            "1. Choose a folder containing uncropped photos.\n"
            "2. Choose a folder to save the cropped photos to.\n"
            "Once those are chosen, it will crop and save the photos to the given folder.\n"
            "Then, the program will ask you to:\n"
            "3. Choose another folder to save a csv to.\n"
            "4. Review the cropped photos for accuracy.\n"
            "Have you read and understood these instructions? y/n ")
while ans.strip() != 'y':
    ans = input("\nPlease review the instructions or type 'exit' to exit the program ")
    if ans.strip() == "exit":
        exit(1)


root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
print("Choose input (un-cropped pictures) folder.")
input_dir = Path(askdirectory())
print("Choose output (where to save cropped pictures) folder.")
output_dir = Path(askdirectory())
output_dir.mkdir(parents=True, exist_ok=True)


# Facial Recognition model
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
    scale_h = 2.3
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
try:
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

except Exception as e:
    print(e)

print(f"Finished cropping photos. Processed: {processed}, Skipped: {skipped}")
print(f"Skipped files: {skip_list}")

try:
    print(f"Continuing to review in 5...")
    for i in range(4,0,-1):
        time.sleep(1)
        print(f"...{i}")
    print("Choose directory to save review decisions.")
    decision = Path(askdirectory())
    print("Initializing comparison.")
    compare_pics.comparison_tool(original_dir=input_dir, crops_dir=output_dir, decision_dir=decision)

except Exception as e:
    print(e)
