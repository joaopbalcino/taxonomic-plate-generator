# taxonomic-plate-generator

Python pipelines for building publication-ready taxonomic figure plates from microscopy images (SEM, light microscopy, etc.). Designed for researchers who need to compose multi-panel figures that meet journal submission standards.

---

## Scripts

### `make_plate.py` — Multi-image plate composer

Composes multiple separate images (e.g. from different specimens or views) into a single plate with a large hero image and a surrounding grid of smaller panels.

**Features:**
- Interactive selection of hero image and grid panels
- Flexible grid layout (user-defined columns × rows)
- Center-crop resizing for best framing
- Optional global contrast normalisation across all panels
- Interactive scale bar calibration per cell (enter original SEM scale bar in µm and px)
- Output at publication resolution (default 300 DPI, 190 mm wide = 2244 px)

---

### `make_plate_amoeba.py` — Single-image plate with interactive ROI selection

Designed for SEM images where you want to highlight sub-regions of a single image — for example, the individual building-block tiles used in the shell construction of testate amoebae (Arcellinida). Takes one SEM image and lets you draw crop regions interactively.

**Features:**
- Hero panel = full SEM image (label A)
- Tile panels = user-defined crops drawn interactively in a matplotlib window (labels B, C, D…)
- Click-and-drag ROI selection with visual feedback; previously confirmed ROIs shown in colour
- Press **Enter** to confirm each ROI, **Esc** to redo the previous one
- Same contrast normalisation and scale bar calibration workflow as `make_plate.py`
- Scale factor calculated independently for hero (full image → resized) and each tile (ROI crop → resized)

---

## Output specifications

Both scripts produce TIFF files (LZW compressed) at:

| Setting | Default | Pixels |
|---|---|---|
| DPI | 300 | — |
| Full page width | 190 mm | 2244 px |
| Single column width | 90 mm | 1063 px |

This meets the minimum requirements for halftone/colour photographs at most journals (≥ 300 DPI; full page ≥ 2244 px, single column ≥ 1063 px).

---

## Requirements

```
Python 3.8+
Pillow
numpy
matplotlib
```

Install dependencies:

```bash
pip install Pillow numpy matplotlib
```

---

## Usage

### `make_plate.py`

```bash
python3 make_plate.py <input_directory> <output_file>
```

Example:

```bash
python3 make_plate.py ./sem_images/ plate_output.tif
```

The script will guide you through image selection, grid layout, contrast settings, and scale bar calibration interactively in the terminal.

---

### `make_plate_amoeba.py`

```bash
python3 make_plate_amoeba.py <image_path> <output_file>
```

Example:

```bash
python3 make_plate_amoeba.py apodera_vas.tif plate_amoeba.tif
```

A matplotlib window will open for each ROI. Drag a rectangle over the region you want to highlight, then press **Enter** to confirm. After all ROIs are selected, the script returns to the terminal for contrast and scale bar settings.

---

## Example plate layout

```
┌────────────────────┬──────┬──────┐
│                    │  B   │  C   │
│    A  (hero)       ├──────┼──────┤
│                    │  D   │  E   │
└────────────────────┴──────┴──────┘
```

Panel letters are added automatically (white text, black outline, top-left corner). Scale bars are drawn in the bottom-right corner of each panel.

---

## Development and attribution

These scripts were developed with assistance from AI coding assistants ([Anthropic Claude](https://www.anthropic.com) and [DeepSeek](https://www.deepseek.com)). All code was reviewed and tested by the authors.

If you use these tools in a publication, please cite or acknowledge accordingly per your journal's guidelines on software and AI tool disclosure.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).  
You are free to use, modify, and distribute this software under the same terms.

---

## Contributing

Bug reports, suggestions, and pull requests are welcome. Please open an issue to discuss changes before submitting a PR.
