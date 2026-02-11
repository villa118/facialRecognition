"""
Review tool: show original + cropped image and let user mark acceptable or not.

Inputs:
- CROPS_DIR: directory containing cropped images (from the cropper script)
- ORIGINALS_DIR: directory containing original images

How it matches files:
- Assumes cropped filenames end with "_head" before the extension,
  e.g. "person1_head.jpg" corresponds to original "person1.jpg".
- Preserves relative paths under the root directories.

Controls:
- a: accept
- r: reject
- s: skip
- q or ESC: quit

Outputs:
- review_results.csv written to OUTPUT_DIR with columns:
  cropped_path, original_path, decision
"""

from pathlib import Path
import csv
import cv2


# ----------------------------
# User input (only what is needed)
# ----------------------------

ORIGINALS_DIR = Path(input("Enter ORIGINALS directory path: ").strip().strip('"'))
CROPS_DIR = Path(input("Enter CROPS directory path: ").strip().strip('"'))
OUTPUT_DIR = Path(input("Enter OUTPUT directory path for review_results.csv: ").strip().strip('"'))

if not ORIGINALS_DIR.is_dir():
    raise ValueError("ORIGINALS_DIR does not exist or is not a directory.")
if not CROPS_DIR.is_dir():
    raise ValueError("CROPS_DIR does not exist or is not a directory.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "review_results.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_original_from_cropped(cropped_path: Path) -> Path | None:
    """
    Map:
      CROPS_DIR/sub/abc_head.jpg -> ORIGINALS_DIR/sub/abc.jpg
    """
    rel = cropped_path.relative_to(CROPS_DIR)
    stem = rel.stem
    if stem.endswith("_crop"):
        orig_stem = stem[:-5]
    else:
        orig_stem = stem  # fallback
    candidate = (ORIGINALS_DIR / rel.parent / (orig_stem + rel.suffix))
    return candidate if candidate.exists() else None


def resize_to_fit(img, max_w=1400, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_side_by_side(original, cropped):
    """
    Make a side-by-side view.
    - Resizes both to same height
    - Adds labels
    """
    # Resize to fit first
    original = resize_to_fit(original)
    cropped = resize_to_fit(cropped)

    oh, ow = original.shape[:2]
    ch, cw = cropped.shape[:2]

    target_h = min(oh, ch)

    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

    original2 = resize_to_height(original, target_h)
    cropped2 = resize_to_height(cropped, target_h)

    # Convert to 3-channel if needed
    if len(original2.shape) == 2:
        original2 = cv2.cvtColor(original2, cv2.COLOR_GRAY2BGR)
    if len(cropped2.shape) == 2:
        cropped2 = cv2.cvtColor(cropped2, cv2.COLOR_GRAY2BGR)

    # Add small padding between
    pad = 10
    spacer = 255 * (original2[:, :pad] * 0) + 255  # white spacer
    side = cv2.hconcat([original2, spacer, cropped2])

    # Labels
    cv2.putText(side, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    x_cropped_label = original2.shape[1] + pad + 10
    cv2.putText(side, "CROPPED", (x_cropped_label, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Footer instructions
    footer = "a=accept   r=reject   s=skip   q/esc=quit"
    cv2.putText(side, footer, (10, side.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return side


def iter_crops(crops_root: Path):
    for p in crops_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main():
    rows = []

    crop_files = list(iter_crops(CROPS_DIR))
    if not crop_files:
        raise FileNotFoundError("No cropped images found in CROPS_DIR.")

    # If CSV already exists, load previous decisions so we can resume
    decided = {}
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                decided[r["cropped_path"]] = r["decision"]

    for cropped_path in crop_files:
        cropped_key = str(cropped_path.relative_to(CROPS_DIR))

        # Skip already decided
        if cropped_key in decided:
            continue

        original_path = find_original_from_cropped(cropped_path)
        if original_path is None:
            # record missing original
            rows.append({
                "cropped_path": cropped_key,
                "original_path": "",
                "decision": "missing_original",
            })
            continue

        orig = cv2.imread(str(original_path))
        crop = cv2.imread(str(cropped_path))
        if orig is None or crop is None:
            rows.append({
                "cropped_path": cropped_key,
                "original_path": str(original_path.relative_to(ORIGINALS_DIR)),
                "decision": "read_error",
            })
            continue

        view = make_side_by_side(orig, crop)
        cv2.imshow("Review (Original | Cropped)", view)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (27, ord("q")):  # ESC or q
                cv2.destroyAllWindows()
                # write any buffered rows and exit
                write_results(rows, CSV_PATH, decided)
                return
            if k == ord("a"):
                rows.append({
                    "cropped_path": cropped_key,
                    "original_path": str(original_path.relative_to(ORIGINALS_DIR)),
                    "decision": "accept",
                })
                break
            if k == ord("r"):
                rows.append({
                    "cropped_path": cropped_key,
                    "original_path": str(original_path.relative_to(ORIGINALS_DIR)),
                    "decision": "reject",
                })
                break
            if k == ord("s"):
                rows.append({
                    "cropped_path": cropped_key,
                    "original_path": str(original_path.relative_to(ORIGINALS_DIR)),
                    "decision": "skip",
                })
                break

        cv2.destroyWindow("Review (Original | Cropped)")

        # Periodically flush to disk (so you can stop any time)
        if len(rows) >= 25:
            write_results(rows, CSV_PATH, decided)
            rows.clear()

    cv2.destroyAllWindows()
    write_results(rows, CSV_PATH, decided)


def write_results(new_rows, csv_path: Path, decided_dict):
    # Merge with already-decided rows
    for r in new_rows:
        decided_dict[r["cropped_path"]] = r["decision"]

    # If file exists, we will rebuild it from prior + new (simple, robust)
    all_rows = []

    # Load existing
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append(r)

    # Add new rows (avoid duplicates)
    existing_keys = {r["cropped_path"] for r in all_rows}
    for r in new_rows:
        if r["cropped_path"] not in existing_keys:
            all_rows.append(r)

    # Write
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["cropped_path", "original_path", "decision"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} decisions to: {csv_path}")


if __name__ == "__main__":
    main()
