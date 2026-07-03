#!/usr/bin/env python3
"""
Interactive Taxonomic Plate Generator — Scale Bars Added After

Builds a publication-ready figure plate:
  1. Compose images into plate (center-crop for best framing)
  2. Interactively calibrate each cell and add fresh scale bars

Usage:
    python3 make_plate_interactive.py [input_dir] [output_file]
"""

import sys
import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_DPI = 300
DEFAULT_WIDTH_MM = 190
DEFAULT_HERO_FRAC = 0.55
BG_COLOR = (255, 255, 255)
COL_GAP = 2
ROW_GAP = 2
LABEL_FONT_FRAC = 0.09
LABEL_OUTLINE = 3
LABEL_MARGIN = 0.03

# Scale bar visual settings
SB_MARGIN = 12          # px from cell edges
SB_HEIGHT = 6           # thickness of bar (px)
SB_OUTLINE = 1          # black outline (px)
SB_FONT_FRAC = 0.07     # font size relative to cell height


def get_font(size):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/helvetica/Helvetica-Bold.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
        "/usr/share/fonts/truetype/urw-base35/NimbusSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ask_choice(prompt, options, allow_none=False):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    if allow_none:
        print("  [0] None / Skip")
    while True:
        try:
            choice = int(input("  Choice: ").strip())
            if allow_none and choice == 0:
                return None
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"  Please enter a number between {'0' if allow_none else '1'} and {len(options)}")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def ask_int(prompt, default=None, min_val=1):
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{default_str}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if val < min_val:
                print(f"  Please enter a number >= {min_val}")
                continue
            return val
        except ValueError:
            print("  Invalid input. Please enter a whole number.")


def ask_float(prompt, default=None, min_val=0.0, allow_empty=False):
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{default_str}: ").strip()
        if raw == "" and allow_empty:
            return default
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if val <= min_val:
                print(f"  Please enter a number > {min_val}")
                continue
            return val
        except ValueError:
            print("  Invalid input. Please enter a number.")


def ask_yes_no(prompt, default=True):
    default_str = " [Y/n]" if default else " [y/N]"
    while True:
        raw = input(f"{prompt}{default_str}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def number_to_letter(n):
    return chr(ord('A') + n - 1)


def discover_images(input_dir):
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    files = []
    for ext in exts:
        files.extend(sorted(Path(input_dir).glob(f"*{ext}")))
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def draw_label(img, letter, cell_w, cell_h):
    draw = ImageDraw.Draw(img)
    fsize = max(16, int(min(cell_w, cell_h) * LABEL_FONT_FRAC))
    font = get_font(fsize)
    ow = LABEL_OUTLINE
    marg = max(8, int(cell_w * LABEL_MARGIN))

    x, y = marg, marg
    for dx in range(-ow, ow + 1):
        for dy in range(-ow, ow + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), letter, font=font, fill="black")
    draw.text((x, y), letter, font=font, fill="white")
    return img


def compute_global_stretch(img_paths, low_pct=1.0, high_pct=99.0):
    """
    Pool a sample of pixels from all images and compute global
    percentile-based stretch limits (p_low, p_high).

    SEM images are grayscale but loaded as RGB — we use the luminance
    (mean of channels) to build one unified histogram across the whole plate.
    Returns (p_low, p_high) as float32 values in [0, 255].
    """
    print("\n  Computing global contrast normalization...")
    all_pixels = []
    for p in img_paths:
        with Image.open(p) as im:
            gray = np.array(im.convert("L"), dtype=np.float32)
        # Subsample to keep memory reasonable (every 4th pixel)
        all_pixels.append(gray[::4, ::4].ravel())
    combined = np.concatenate(all_pixels)
    p_low  = np.percentile(combined, low_pct)
    p_high = np.percentile(combined, high_pct)
    print(f"  Global stretch range: {p_low:.1f} – {p_high:.1f}  "
          f"(p{low_pct} – p{high_pct} across all images)")
    return float(p_low), float(p_high)


def apply_global_stretch(img, p_low, p_high):
    """
    Apply the same linear stretch to an RGB image:
      - pixels at p_low  → 0
      - pixels at p_high → 255
    Channels are stretched together so grayscale SEM images stay neutral.
    """
    arr = np.array(img, dtype=np.float32)
    arr = (arr - p_low) / max(p_high - p_low, 1e-6) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def resize_exact(img, w, h):
    """Center-crop to w×h — best composition."""
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    nw = round(img.width * scale)
    nh = round(img.height * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def draw_scalebar_on_cell(cell_img, target_um, px_per_um):
    """
    Draw a scale bar on the bottom-right of a cell image.
    target_um: desired length of scale bar in µm
    px_per_um: calibration from original image (already scaled to cell size)
    """
    draw = ImageDraw.Draw(cell_img)
    W, H = cell_img.size

    # True physical bar length — never inflate this
    bar_px = round(target_um * px_per_um)

    # Font size: relative to the SHORTER side so hero and small cells are consistent.
    # Also cap to a fraction of the bar's own pixel width so it never dwarfs the bar.
    short_side = min(W, H)
    fsize_candidate = int(short_side * SB_FONT_FRAC)
    fsize_max_from_bar = max(40, bar_px // 3)
    fsize = max(10, min(fsize_candidate, fsize_max_from_bar))
    font = get_font(fsize)

    label = f"{target_um:g} \u00b5m"

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Shrink font until label fits within bar width
    while text_w > bar_px and fsize > 10:
        fsize -= 1
        font = get_font(fsize)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    bar_w = bar_px  # authoritative — never widened to match text

    # Position: bottom-right, then clamp everything inside the cell
    margin = max(SB_MARGIN, SB_OUTLINE + 1)
    bar_x2 = W - margin
    bar_x1 = bar_x2 - bar_w
    bar_bottom = H - margin
    bar_top = bar_bottom - SB_HEIGHT

    if bar_x1 < margin:
        bar_x1 = margin
        bar_x2 = bar_x1 + bar_w
    bar_x2 = min(bar_x2, W - margin)

    label_y = max(margin, bar_top - text_h - 3)
    label_x = bar_x1 + (bar_w - text_w) // 2
    label_x = max(margin, min(label_x, W - margin - text_w))

    # Draw label with black outline
    for dx in range(-SB_OUTLINE, SB_OUTLINE + 1):
        for dy in range(-SB_OUTLINE, SB_OUTLINE + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((label_x + dx, label_y + dy), label, font=font, fill="black")
    draw.text((label_x, label_y), label, font=font, fill="white")

    # Draw bar: black outline then white fill
    draw.rectangle(
        [bar_x1 - SB_OUTLINE, bar_top - SB_OUTLINE,
         bar_x2 + SB_OUTLINE, bar_bottom + SB_OUTLINE],
        fill="black"
    )
    draw.rectangle([bar_x1, bar_top, bar_x2, bar_bottom], fill="white")

    return cell_img


def build_plate_interactive(input_dir, output_path):
    # ── 1. Discover images ───────────────────────────────────────────────
    images = discover_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Found {len(images)} image(s) in: {input_dir}")
    print(f"{'='*60}")
    for i, img in enumerate(images, 1):
        with Image.open(img) as im:
            mode = "portrait" if im.height > im.width else "landscape"
            print(f"  [{i}] {img.name}  ({im.width}×{im.height}, {mode})")
    
    # ── 2. Select hero image ─────────────────────────────────────────────
    hero = ask_choice("Which image should be the large hero (main image)?", images)
    hero_idx = images.index(hero) + 1
    
    # ── 3. Select surrounding images ─────────────────────────────────────
    remaining = [img for img in images if img != hero]
    if not remaining:
        print("No other images available for the grid.")
        sys.exit(1)
    
    print(f"\nRemaining images available for surrounding grid:")
    for i, img in enumerate(remaining, 1):
        with Image.open(img) as im:
            mode = "portrait" if im.height > im.width else "landscape"
            print(f"  [{i}] {img.name}  ({im.width}×{im.height}, {mode})")
    
    use_all = ask_yes_no("Use all remaining images in the grid?", default=True)
    if use_all:
        small_images = remaining
    else:
        small_images = []
        print("Select images for the grid (enter number, 0 to finish):")
        available = remaining.copy()
        while available:
            for i, img in enumerate(available, 1):
                print(f"    [{i}] {img.name}")
            print("    [0] Done selecting")
            choice = ask_int("  Add image", min_val=0)
            if choice == 0:
                break
            if 1 <= choice <= len(available):
                small_images.append(available.pop(choice - 1))
            else:
                print("  Invalid choice.")
        if not small_images:
            print("No images selected for grid.")
            sys.exit(1)
    
    n_small = len(small_images)
    print(f"\nSelected {n_small} image(s) for the surrounding grid.")
    
    # ── 4. Choose grid layout ────────────────────────────────────────────
    print(f"\nSuggested grid layouts for {n_small} images:")
    layouts = []
    for cols in range(1, min(n_small + 1, 6)):
        rows = math.ceil(n_small / cols)
        layouts.append((cols, rows))
    
    for i, (cols, rows) in enumerate(layouts, 1):
        print(f"  [{i}] {cols} columns × {rows} rows")
    
    n_cols = ask_int("How many columns for the small grid?", default=layouts[0][0] if layouts else 2)
    n_rows = math.ceil(n_small / n_cols)
    print(f"  → Grid will be {n_cols} columns × {n_rows} rows")
    
    # ── 5. Output settings ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Output settings")
    print(f"{'='*60}")
    
    width_mm = ask_float("Plate width (mm)", default=DEFAULT_WIDTH_MM)
    dpi = ask_int("DPI", default=DEFAULT_DPI)
    plate_w = round(width_mm / 25.4 * dpi)
    
    hero_frac = ask_float("Hero width as fraction of total width (0.3-0.7)", 
                          default=DEFAULT_HERO_FRAC, min_val=0.1)
    hero_frac = max(0.2, min(0.8, hero_frac))
    
    # ── 6. Calculate geometry ────────────────────────────────────────────
    with Image.open(hero) as im:
        hero_aspect = im.height / im.width
    
    hero_w = round(plate_w * hero_frac)
    remaining_w = plate_w - hero_w - COL_GAP
    small_w = (remaining_w - (n_cols - 1) * COL_GAP) // n_cols
    hero_w = plate_w - (n_cols * small_w + (n_cols - 1) * COL_GAP) - COL_GAP
    
    hero_h = round(hero_w * hero_aspect)
    small_h = (hero_h - (n_rows - 1) * ROW_GAP) // n_rows
    hero_h = n_rows * small_h + (n_rows - 1) * ROW_GAP
    plate_h = hero_h
    
    print(f"\n{'='*60}")
    print("Layout geometry:")
    print(f"{'='*60}")
    print(f"  Plate:     {plate_w} × {plate_h} px  ({plate_w/dpi*25.4:.1f} × {plate_h/dpi*25.4:.1f} mm)")
    print(f"  Hero:      {hero_w} × {hero_h} px  (image {hero_idx})")
    print(f"  Small:     {small_w} × {small_h} px each")
    print(f"  Grid:      {n_cols} cols × {n_rows} rows")
    print(f"  Gaps:      {COL_GAP}px horizontal, {ROW_GAP}px vertical")
    
    # ── 7. Build plate (composition only) ────────────────────────────────

    # Global contrast normalization (optional)
    do_norm = ask_yes_no("\nNormalize contrast/brightness globally across all images?",
                         default=True)
    p_low, p_high = 0.0, 255.0  # identity stretch (no-op) if user declines
    if do_norm:
        low_pct  = ask_float("  Shadow clip percentile  (default 1.0)", default=1.0,
                             min_val=0.0, allow_empty=True)
        high_pct = ask_float("  Highlight clip percentile (default 99.0)", default=99.0,
                             min_val=0.0, allow_empty=True)
        all_img_paths = [hero] + list(small_images)
        p_low, p_high = compute_global_stretch(all_img_paths,
                                               low_pct=low_pct, high_pct=high_pct)

    plate = Image.new("RGB", (plate_w, plate_h), BG_COLOR)
    cells_info = []  # track (x, y, w, h, img_path, letter) for scale bar step

    def paste_image(img_path, x, y, w, h, label_num):
        cell = resize_exact(Image.open(img_path), w, h)
        if do_norm:
            cell = apply_global_stretch(cell, p_low, p_high)
        letter = number_to_letter(label_num)
        cell = draw_label(cell, letter, w, h)
        plate.paste(cell, (x, y))
        cells_info.append((x, y, w, h, img_path, letter))
        print(f"  {label_num:2d}. {img_path.name} → {letter}  ({x},{y}) {w}×{h}")
        return label_num + 1
    
    label_num = 1
    
    print(f"\nBuilding plate composition...")
    label_num = paste_image(hero, 0, 0, hero_w, hero_h, label_num)
    
    small_col_xs = [hero_w + COL_GAP + i * (small_w + COL_GAP) for i in range(n_cols)]
    small_row_ys = [j * (small_h + ROW_GAP) for j in range(n_rows)]
    
    for idx, img_path in enumerate(small_images):
        row, col = divmod(idx, n_cols)
        if row >= n_rows:
            break
        x = small_col_xs[col]
        y = small_row_ys[row]
        label_num = paste_image(img_path, x, y, small_w, small_h, label_num)
    
    # ── 8. Add scale bars interactively ──────────────────────────────────
    print(f"\n{'='*60}")
    print("SCALE BAR CALIBRATION")
    print(f"{'='*60}")
    print("For each cell, you'll enter the original SEM calibration.")
    print("Then I'll draw a fresh scale bar on the composed plate.")
    print(f"{'='*60}")
    
    # Ask for default scale bar length
    default_target_um = ask_float("Default scale bar length in µm (press Enter for 10)", 
                                   default=10.0, allow_empty=True)
    
    for i, (x, y, w, h, img_path, letter) in enumerate(cells_info):
        print(f"\n  Cell {letter}: {img_path.name}  ({w}×{h} px)")
        
        ref_um = ask_float("    Original SEM scale bar length (µm): ", min_val=0.001)
        ref_px = ask_float("    Original SEM scale bar length (pixels): ", min_val=0.001)
        
        # Original px/µm ratio
        px_per_um_original = ref_px / ref_um
        
        # Open once with RGB conversion — matches resize_exact exactly,
        # resolving EXIF orientation so dimensions are always correct.
        with Image.open(img_path) as orig:
            orig_rgb = orig.convert("RGB")
            orig_w, orig_h = orig_rgb.size
        scale_factor = max(w / orig_w, h / orig_h)
        px_per_um_cell = px_per_um_original * scale_factor
        
        # Ask for target scale bar length for this cell
        target_um = ask_float(f"    Desired scale bar length in µm (Enter for {default_target_um})", 
                              default=default_target_um, allow_empty=True)
        
        # Draw scale bar directly on the plate at this cell's position
        # Extract cell region, draw bar, paste back
        cell_region = plate.crop((x, y, x + w, y + h))
        cell_region = draw_scalebar_on_cell(cell_region, target_um, px_per_um_cell)
        plate.paste(cell_region, (x, y))
        
        bar_px = round(target_um * px_per_um_cell)
        print(f"    → Scale bar: {target_um:g} µm = {bar_px} px")
    
    # ── 9. Save ──────────────────────────────────────────────────────────
    plate.save(output_path, dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"\n{'='*60}")
    print(f"✓ Saved: {output_path}")
    print(f"  Size: {plate_w}×{plate_h} px @ {dpi} DPI")
    print(f"  Print: {plate_w/dpi*25.4:.1f}×{plate_h/dpi*25.4:.1f} mm")
    print(f"{'='*60}")


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_path = sys.argv[2] if len(sys.argv) > 2 else "plate.tif"
    
    if not Path(input_dir).is_dir():
        print(f"Not a directory: {input_dir}")
        sys.exit(1)
    
    build_plate_interactive(input_dir, output_path)


if __name__ == "__main__":
    main()
