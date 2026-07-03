#!/usr/bin/env python3
"""
Taxonomic Plate Generator — Single SEM Image with Interactive ROI Selection
For Arcellinida (testate amoeba) SEM plates.

Workflow:
  1. Open the SEM image in an interactive matplotlib window
  2. Click-and-drag to draw each ROI rectangle (building-block tiles)
  3. Press Enter to confirm each ROI and move to the next
  4. Plate is built: hero (full image) on the left, N×2 grid of crops on the right
  5. Interactive scale bar calibration (same as make_plate.py)

Usage:
    python3 make_plate_amoeba.py <image_path> [output_file]

    e.g.:
    python3 make_plate_amoeba.py apodera_vas.tif plate_amoeba.tif
"""

import sys
import os
import math
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageDraw, ImageFont

# ── Layout defaults ──────────────────────────────────────────────────────────
DEFAULT_DPI       = 300
DEFAULT_WIDTH_MM  = 190
DEFAULT_HERO_FRAC = 0.55   # hero width as fraction of total plate width
BG_COLOR          = (255, 255, 255)
COL_GAP           = 2      # px gap between hero and grid, and between grid cells
ROW_GAP           = 2

# Label settings
LABEL_FONT_FRAC = 0.09
LABEL_OUTLINE   = 3
LABEL_MARGIN    = 0.03

# Scale bar settings
SB_MARGIN     = 12
SB_HEIGHT     = 6
SB_OUTLINE    = 1
SB_FONT_FRAC  = 0.07

# ROI selector colours (cycling, for visual feedback during selection)
ROI_COLORS = ["#FF4444", "#44AAFF", "#44DD88", "#FFAA00",
              "#CC44FF", "#FF88CC", "#00DDDD", "#FFFF44"]


# ── Helpers (identical to make_plate.py) ────────────────────────────────────

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
        if (raw == "") and (allow_empty or default is not None):
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


def draw_label(img, letter, cell_w, cell_h):
    draw = ImageDraw.Draw(img)
    fsize = max(16, int(min(cell_w, cell_h) * LABEL_FONT_FRAC))
    font  = get_font(fsize)
    ow    = LABEL_OUTLINE
    marg  = max(8, int(cell_w * LABEL_MARGIN))
    x, y  = marg, marg
    for dx in range(-ow, ow + 1):
        for dy in range(-ow, ow + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), letter, font=font, fill="black")
    draw.text((x, y), letter, font=font, fill="white")
    return img


def resize_exact(img, w, h):
    """Center-crop to w×h — best composition."""
    img   = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    nw    = round(img.width  * scale)
    nh    = round(img.height * scale)
    img   = img.resize((nw, nh), Image.LANCZOS)
    left  = (nw - w) // 2
    top   = (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def crop_exact(img, x, y, w, h):
    """Crop a region from img at pixel coords (x, y, w, h)."""
    img = img.convert("RGB")
    return img.crop((x, y, x + w, y + h))


def compute_global_stretch(pil_images, low_pct=1.0, high_pct=99.0):
    print("\n  Computing global contrast normalisation...")
    all_pixels = []
    for im in pil_images:
        gray = np.array(im.convert("L"), dtype=np.float32)
        all_pixels.append(gray[::4, ::4].ravel())
    combined = np.concatenate(all_pixels)
    p_low  = np.percentile(combined, low_pct)
    p_high = np.percentile(combined, high_pct)
    print(f"  Global stretch range: {p_low:.1f} – {p_high:.1f}  "
          f"(p{low_pct} – p{high_pct})")
    return float(p_low), float(p_high)


def apply_global_stretch(img, p_low, p_high):
    arr = np.array(img, dtype=np.float32)
    arr = (arr - p_low) / max(p_high - p_low, 1e-6) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def draw_scalebar_on_cell(cell_img, target_um, px_per_um):
    draw  = ImageDraw.Draw(cell_img)
    W, H  = cell_img.size
    bar_px = round(target_um * px_per_um)

    short_side       = min(W, H)
    fsize_candidate  = int(short_side * SB_FONT_FRAC)
    fsize_max_from_bar = max(40, bar_px // 3)
    fsize = max(10, min(fsize_candidate, fsize_max_from_bar))
    font  = get_font(fsize)

    label = f"{target_um:g} µm"
    bbox  = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    while text_w > bar_px and fsize > 10:
        fsize -= 1
        font   = get_font(fsize)
        bbox   = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    bar_w   = bar_px
    margin  = max(SB_MARGIN, SB_OUTLINE + 1)
    bar_x2  = W - margin
    bar_x1  = bar_x2 - bar_w
    bar_bottom = H - margin
    bar_top    = bar_bottom - SB_HEIGHT

    if bar_x1 < margin:
        bar_x1 = margin
        bar_x2 = bar_x1 + bar_w
    bar_x2 = min(bar_x2, W - margin)

    label_y = max(margin, bar_top - text_h - 3)
    label_x = bar_x1 + (bar_w - text_w) // 2
    label_x = max(margin, min(label_x, W - margin - text_w))

    for dx in range(-SB_OUTLINE, SB_OUTLINE + 1):
        for dy in range(-SB_OUTLINE, SB_OUTLINE + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((label_x + dx, label_y + dy), label, font=font, fill="black")
    draw.text((label_x, label_y), label, font=font, fill="white")

    draw.rectangle(
        [bar_x1 - SB_OUTLINE, bar_top - SB_OUTLINE,
         bar_x2 + SB_OUTLINE, bar_bottom + SB_OUTLINE],
        fill="black"
    )
    draw.rectangle([bar_x1, bar_top, bar_x2, bar_bottom], fill="white")
    return cell_img


# ── Interactive ROI selection ────────────────────────────────────────────────

def select_rois_interactively(image_path, n_rois):
    """
    Open the SEM image in a matplotlib window.
    User draws n_rois rectangles with click-and-drag.
    Returns list of (x, y, w, h) in original image pixel coordinates.
    """
    with Image.open(image_path) as im:
        img_array = np.array(im.convert("RGB"))
    orig_h, orig_w = img_array.shape[:2]

    rois      = []        # confirmed ROIs as (x, y, w, h) in original px
    current   = [None]    # holds the latest selector rectangle (extents)
    confirmed = [False]   # flag: user pressed Enter on current ROI

    def on_select(eclick, erelease):
        """Called while dragging — stores the current rectangle."""
        x0, y0 = eclick.xdata,   eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        current[0] = (min(x0, x1), min(y0, y1),
                      max(x0, x1), max(y0, y1))   # display coords

    def on_key(event):
        """Enter = confirm current ROI; Escape = redo last."""
        if event.key == "enter":
            if current[0] is None:
                print("  Draw a rectangle first, then press Enter.")
                return
            confirmed[0] = True
            plt.close()

        elif event.key == "escape":
            if rois:
                removed = rois.pop()
                print(f"  Removed last ROI: {removed}. Re-draw it.")
            current[0] = None
            confirmed[0] = False
            plt.close()

    for roi_idx in range(n_rois):
        current[0]   = None
        confirmed[0] = False

        fig, ax = plt.subplots(figsize=(10, 12))
        fig.canvas.manager.set_window_title(
            f"ROI {roi_idx + 1} of {n_rois}  —  drag a rectangle, press Enter to confirm"
        )
        ax.imshow(img_array, cmap="gray" if img_array.ndim == 2 else None,
                  interpolation="nearest")
        ax.set_title(
            f"Select ROI {number_to_letter(roi_idx + 2)} "   # hero = A, crops = B, C …
            f"({roi_idx + 1} of {n_rois})\n"
            f"Click-and-drag to draw rectangle  •  Enter to confirm  •  Esc to redo previous",
            fontsize=10
        )

        # Draw already-confirmed ROIs in their colours
        for i, (rx, ry, rw, rh) in enumerate(rois):
            col  = ROI_COLORS[i % len(ROI_COLORS)]
            rect = mpatches.Rectangle(
                (rx, ry), rw, rh,
                linewidth=2, edgecolor=col, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(rx + 4, ry + 4,
                    number_to_letter(i + 2),   # B, C, D …
                    color=col, fontsize=11,
                    fontweight="bold",
                    va="top", ha="left")

        col_current = ROI_COLORS[roi_idx % len(ROI_COLORS)]
        selector = RectangleSelector(
            ax, on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(edgecolor=col_current, facecolor="none", linewidth=2)
        )

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()

        print(f"\n  [ROI {roi_idx + 1}/{n_rois}]  Drag a rectangle over a building-block tile.")
        print("  Press Enter to confirm, Esc to undo the previous ROI.")

        plt.show()   # blocks until user closes window (via Enter/Esc handler)

        if not confirmed[0]:
            # Window was closed without confirmation — treat as skip
            print("  Window closed without confirmation — skipping remaining ROIs.")
            break

        if current[0] is None:
            print("  No rectangle drawn — skipping this ROI.")
            continue

        # Convert display (float) coords to integer pixel coords in original image
        dx0, dy0, dx1, dy1 = current[0]
        # Clamp to image bounds
        dx0 = max(0, int(round(dx0)))
        dy0 = max(0, int(round(dy0)))
        dx1 = min(orig_w, int(round(dx1)))
        dy1 = min(orig_h, int(round(dy1)))
        rx, ry, rw, rh = dx0, dy0, dx1 - dx0, dy1 - dy0
        rois.append((rx, ry, rw, rh))
        print(f"  ✓ ROI {number_to_letter(roi_idx + 2)} confirmed: "
              f"x={rx}, y={ry}, w={rw}, h={rh}")

    return rois


# ── Main plate builder ───────────────────────────────────────────────────────

def build_plate(image_path, output_path):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    with Image.open(image_path) as im:
        orig_w, orig_h = im.size
        mode = "portrait" if orig_h > orig_w else "landscape"

    print(f"\n{'='*60}")
    print(f"  Image : {image_path.name}  ({orig_w}×{orig_h}, {mode})")
    print(f"{'='*60}")

    # ── 1. How many ROIs? ────────────────────────────────────────────────
    n_rois = ask_int("How many building-block tiles to highlight?", default=4)

    # ── 2. Grid layout ───────────────────────────────────────────────────
    print(f"\nSuggested grid layouts for {n_rois} tiles:")
    layouts = []
    for cols in range(1, min(n_rois + 1, 6)):
        rows = math.ceil(n_rois / cols)
        layouts.append((cols, rows))
    for i, (c, r) in enumerate(layouts, 1):
        print(f"  [{i}] {c} columns × {r} rows")
    n_cols = ask_int("How many columns for the tile grid?",
                     default=min(2, n_rois))
    n_rows = math.ceil(n_rois / n_cols)
    print(f"  → Grid: {n_cols} cols × {n_rows} rows")

    # ── 3. Output settings ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Output settings")
    print(f"{'='*60}")
    width_mm  = ask_float("Plate width (mm)", default=DEFAULT_WIDTH_MM)
    dpi       = ask_int("DPI", default=DEFAULT_DPI)
    plate_w   = round(width_mm / 25.4 * dpi)
    hero_frac = ask_float(
        "Hero width as fraction of total plate width (0.3–0.7)",
        default=DEFAULT_HERO_FRAC, min_val=0.1
    )
    hero_frac = max(0.2, min(0.8, hero_frac))

    # ── 4. Geometry ──────────────────────────────────────────────────────
    hero_aspect = orig_h / orig_w   # portrait image

    # Work out cell sizes so everything lines up perfectly
    small_w = (plate_w - round(plate_w * hero_frac) - COL_GAP
               - (n_cols - 1) * COL_GAP) // n_cols
    hero_w  = plate_w - (n_cols * small_w + (n_cols - 1) * COL_GAP) - COL_GAP
    hero_h  = round(hero_w * hero_aspect)
    small_h = (hero_h - (n_rows - 1) * ROW_GAP) // n_rows
    hero_h  = n_rows * small_h + (n_rows - 1) * ROW_GAP
    plate_h = hero_h

    print(f"\n{'='*60}")
    print("Layout geometry:")
    print(f"{'='*60}")
    print(f"  Plate  : {plate_w} × {plate_h} px  "
          f"({plate_w/dpi*25.4:.1f} × {plate_h/dpi*25.4:.1f} mm)")
    print(f"  Hero   : {hero_w} × {hero_h} px")
    print(f"  Tiles  : {small_w} × {small_h} px each")
    print(f"  Grid   : {n_cols} cols × {n_rows} rows")

    # ── 5. Interactive ROI selection ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("INTERACTIVE ROI SELECTION")
    print(f"{'='*60}")
    print("A matplotlib window will open for each tile.")
    print("Drag a rectangle over the building-block you want to highlight.")
    print("Press Enter to confirm each selection.\n")

    rois = select_rois_interactively(image_path, n_rois)

    if not rois:
        print("No ROIs selected — aborting.")
        sys.exit(1)

    if len(rois) < n_rois:
        print(f"\nNote: only {len(rois)} of {n_rois} ROIs were confirmed. "
              f"Continuing with {len(rois)} tiles.")
        n_rois = len(rois)
        n_rows = math.ceil(n_rois / n_cols)

    # ── 6. Contrast normalisation ────────────────────────────────────────
    do_norm = ask_yes_no(
        "\nNormalise contrast/brightness globally across hero + all tiles?",
        default=True
    )
    p_low, p_high = 0.0, 255.0
    if do_norm:
        low_pct  = ask_float("  Shadow clip percentile  (default 1.0)",
                             default=1.0, min_val=0.0, allow_empty=True)
        high_pct = ask_float("  Highlight clip percentile (default 99.0)",
                             default=99.0, min_val=0.0, allow_empty=True)
        with Image.open(image_path) as im:
            hero_pil = im.convert("RGB")
            # Build tile PIL images for pooled histogram
            tile_pils = [crop_exact(hero_pil, *roi) for roi in rois]
        all_pils = [hero_pil] + tile_pils
        p_low, p_high = compute_global_stretch(all_pils,
                                               low_pct=low_pct,
                                               high_pct=high_pct)

    # ── 7. Build plate ───────────────────────────────────────────────────
    print(f"\nBuilding plate…")
    plate      = Image.new("RGB", (plate_w, plate_h), BG_COLOR)
    cells_info = []   # (x, y, w, h, label, source_roi_or_None)

    with Image.open(image_path) as im:
        src = im.convert("RGB")

    # — Hero (label A) —
    hero_cell = resize_exact(src.copy(), hero_w, hero_h)
    if do_norm:
        hero_cell = apply_global_stretch(hero_cell, p_low, p_high)
    hero_cell = draw_label(hero_cell, "A", hero_w, hero_h)
    plate.paste(hero_cell, (0, 0))
    cells_info.append((0, 0, hero_w, hero_h, "A", None))
    print(f"  A  Hero  ({hero_w}×{hero_h} px)")

    # — Tiles (labels B, C, …) —
    small_col_xs = [hero_w + COL_GAP + i * (small_w + COL_GAP)
                    for i in range(n_cols)]
    small_row_ys = [j * (small_h + ROW_GAP) for j in range(n_rows)]

    for idx, roi in enumerate(rois):
        row, col = divmod(idx, n_cols)
        if row >= n_rows:
            break
        rx, ry, rw, rh = roi
        letter = number_to_letter(idx + 2)   # B, C, D …
        x = small_col_xs[col]
        y = small_row_ys[row]

        tile = crop_exact(src, rx, ry, rw, rh)
        tile = resize_exact(tile, small_w, small_h)
        if do_norm:
            tile = apply_global_stretch(tile, p_low, p_high)
        tile = draw_label(tile, letter, small_w, small_h)
        plate.paste(tile, (x, y))
        cells_info.append((x, y, small_w, small_h, letter, roi))
        print(f"  {letter}  ROI x={rx},y={ry} w={rw}×{rh}  →  tile ({x},{y}) {small_w}×{small_h}")

    # ── 8. Scale bars ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SCALE BAR CALIBRATION")
    print(f"{'='*60}")
    print("For each cell enter the SEM calibration from the original image.")
    print(f"{'='*60}")

    default_target_um = ask_float(
        "Default scale bar length in µm [10]",
        default=10.0, allow_empty=True
    )

    for (x, y, w, h, letter, roi) in cells_info:
        print(f"\n  Cell {letter}:  {w}×{h} px on plate")
        ref_um = ask_float("    Original SEM scale bar length (µm): ",
                           min_val=0.001)
        ref_px = ask_float("    Original SEM scale bar length (pixels): ",
                           min_val=0.001)

        px_per_um_orig = ref_px / ref_um

        if roi is None:
            # Hero: scaled from full original image
            scale_factor = max(w / orig_w, h / orig_h)
        else:
            rx, ry, rw, rh = roi
            # Tile: scaled from the cropped ROI
            scale_factor = max(w / rw, h / rh)

        px_per_um_cell = px_per_um_orig * scale_factor

        target_um = ask_float(
            f"    Desired scale bar length in µm [Enter for {default_target_um}]",
            default=default_target_um, allow_empty=True
        )

        cell_region = plate.crop((x, y, x + w, y + h))
        cell_region = draw_scalebar_on_cell(cell_region, target_um, px_per_um_cell)
        plate.paste(cell_region, (x, y))

        bar_px = round(target_um * px_per_um_cell)
        print(f"    → Scale bar: {target_um:g} µm = {bar_px} px on plate")

    # ── 9. Save ──────────────────────────────────────────────────────────
    plate.save(output_path, dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"\n{'='*60}")
    print(f"✓  Saved: {output_path}")
    print(f"   Size : {plate_w}×{plate_h} px @ {dpi} DPI")
    print(f"   Print: {plate_w/dpi*25.4:.1f}×{plate_h/dpi*25.4:.1f} mm")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "plate_amoeba.tif"
    build_plate(image_path, output_path)


if __name__ == "__main__":
    main()
