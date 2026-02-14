"""
Defaults:
- Uses Haar cascade frontal face detector
- Crops largest detected face
- Expands crop to include full head (hair + slight neck)
- Preserves folder structure
"""
import tkinter.messagebox
from pathlib import Path
import cv2
import tkinter as tk
from tkinter.filedialog import askdirectory
import tkinter.messagebox
from tkinter import ttk
import compare_pics
import time

# ----------------------------
# Helper functions
# ----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ask_input_output():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    tkinter.messagebox.showinfo(title="input", message="Choose input (un-cropped pictures) folder.")
    input_dir = Path(askdirectory())
    tkinter.messagebox.showinfo(title="output", message="Choose output (where to save cropped pictures) folder.")
    output_dir = Path(askdirectory())
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def initialize_model():
    # Facial Recognition model
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade.")
    return face_cascade


def display_instructions():
    return tk.messagebox.askyesno("READ INSTRUCTIONS", "This script will ask you to do the following:\n"
                                                       "1. Choose a folder containing un-cropped photos.\n"
                                                       "2. You MUST choose a DIFFERENT folder to save the cropped photos to or it will not work properly.\n"
                                                       "You may create a new folder in the dialogue window if needed.\n"
                                                       "Once those are chosen, it will crop and save the photos to the given folder.\n"
                                                       "Then, the program will ask you to:\n"
                                                       "3. Choose another folder to save a csv to.\n"
                                                       "4. Review the cropped photos for accuracy.\n"
                                                       "Have you read and understood these instructions?")


def create_progress_window(root, total, title="Cropping progress"):
    win = tk.Toplevel(root)
    win.title(title)
    win.resizable(False, False)

    label_var = tk.StringVar(value=f"0 / {total}")

    ttk.Label(win, textvariable=label_var).pack(padx=12, pady=(12, 6))

    bar = ttk.Progressbar(
        win,
        orient="horizontal",
        length=320,
        mode="determinate",
        maximum=total
    )
    bar.pack(padx=12, pady=(0, 12))

    win.update_idletasks()

    return win, bar, label_var

def update_progress(win, bar, label_var, current, total):
    bar["value"] = current
    label_var.set(f"{current} / {total}")
    win.update_idletasks()
    win.update()

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
def process_images(root, input_dir, output_dir):
    processed = 0
    skipped = 0
    skip_list = []

    img_files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    total = len(img_files)

    if total == 0:
        tk.messagebox.showerror("Error", "No images found.")
        return

    win, bar, label_var = create_progress_window(root, total)

    face_cascade = initialize_model()

    try:
        for idx, img_path in enumerate(img_files, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                update_progress(win, bar, label_var, idx, total)
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
                skip_list.append(str(img_path))
                update_progress(win, bar, label_var, idx, total)
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            img_h, img_w = img.shape[:2]
            x1, y1, x2, y2 = expand_to_head(x, y, w, h, img_w, img_h)
            crop = img[y1:y2, x1:x2]

            relative_path = img_path.relative_to(input_dir)
            save_path = (output_dir / relative_path).with_name(
                relative_path.stem + "_crop" + relative_path.suffix
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if cv2.imwrite(str(save_path), crop):
                processed += 1
            else:
                skipped += 1

            update_progress(win, bar, label_var, idx, total)

    finally:
        win.destroy()

    tk.messagebox.showinfo(
        message=f"Finished cropping photos. Processed: {processed}, Skipped: {skipped}"
    )



def review_cropped_image(input_dir, output_dir):
    try:
        tk.messagebox.askokcancel("Review", "When ready to review cropped images, click 'OK'")
        tk.messagebox.showinfo("Choose Folder", "Choose directory to save review decisions.")
        decision = Path(askdirectory())
        tk.messagebox.showinfo("Comparison", "Initializing comparison.")
        compare_pics.comparison_tool(original_dir=input_dir, crops_dir=output_dir, decision_dir=decision)

    except Exception as e:
        tk.messagebox.showerror(str(e))


def main():
    # Get user input

    ans = display_instructions()
    while not ans:
        ans = tk.messagebox.askretrycancel("READ THE INSTRUCTIONS",
                                           "Please review the instructions or exit the program.")
        if not ans:
            exit(1)
        else:
            ans = display_instructions()
    root = tk.Tk()
    root.withdraw()
    input_dir, output_dir = ask_input_output()
    process_images(root, input_dir, output_dir)
    review_cropped_image(input_dir, output_dir)


if __name__ == "__main__":
    main()
