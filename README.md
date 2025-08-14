# Traffic-surveillance

This project provides tools for:
- Detecting lanes in videos or livestreams.
- Measuring vehicle speed in km/h (with required perspective calibration).
- Distance and Traffic analysis.
- Optional super-resolution for enhanced image clarity.

---

## Installation & Setup

### 1. Install Python
- Install **Python 3.10.11** and add it to your system PATH.

### 2. Open a Terminal
- Navigate to the project folder.

### 3. Create a Virtual Environment
```bash
py -3.10 -m venv venv310
```

### 4. Activate the Virtual Environment
```bash
.\venv310\Scripts\activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Configure the Video Path
- In `main.py`, set the path to the video you want to analyze.

### 7. Remove Old Lane Data (if needed)
- If lanes for the selected video have not been defined yet, delete the file:
```
data/lanes.pkl
```
This file stores the last defined lanes only.

### 8. Run the Program
```bash
python main.py
```

---

## Optional Settings

### Enable or Disable Super Resolution
In `main.py`:
```python
use_super_resolution = True  # or False
```

---

## Perspective Calibration (Required for Speed Measurement)

To measure speeds correctly in km/h, you must first perform **perspective calibration**.

### Step-by-Step Calibration
1. Open:
```
speed/calibrate_perspective.py
```
(or your calibration file).

2. Set the video path:
```python
VIDEO_PATH = "path/to/your/video.mp4"
```

3. Run the calibration:
```bash
python speed/calibrate_perspective.py
```

4. Choose the calibration method:
   - **1** = Automatic (Segformer AI-based road detection)
   - **2** = Manual (click 4 points to define road rectangle: front/back left/right lane markers)

5. Save the calibration:
   - The perspective matrix is stored in:
     ```
     KAL/MATFILE.npy
     ```
   - This file will be loaded automatically during analysis.

⚠️ **Important**: Perform calibration before your first analysis. Without it, real speeds (km/h) cannot be calculated.

---

## Lane Definition Guide

When the program starts, the first frame of the video/stream is shown with **green lines** (auto-detected lanes). Selected lanes turn **red**.

### Step 1: Selecting Lane Lines
- Select correct lines in order, starting from **rightmost or leftmost**.
- Select only **one line per lane** (duplicates may exist).
- If there’s an obstacle between lanes (e.g., concrete barrier), press **N** before selecting the next line to start a **new group**.
- Continue selecting until all desired lines are chosen.
- Press **S** to save selection.

### Step 2: Adjusting Lines (Bezier Curves)
- After saving, the line adjustment window opens.
- Each line has:
  - **2 green points** (start & end) – adjust length.
  - **1 red point** (curve control) – adjust curvature.
- When finished, press **Q** to exit calibration.
- The system will switch to traffic analysis with your selected lanes shown as polygons.

**If not all lanes were detected:**
- **Manual Definition immediately**:
  - Press **S** without selecting lines to start manual definition.
- **Partial Selection**:
  - Select detected lines, then press **S** to define missing lines manually.

### Step 3: Manual Lane Definition
- Click **three points** along the desired lane line:
  1. Top of image (furthest point)
  2. Middle
  3. Bottom (closest to camera)
- A Bezier curve will be created, which can be adjusted as in Step 2.

**New Group in Manual Mode**:
- If you don’t want a line to connect with the previous one:
  - Press **N** to enable New Group mode.
  - Define the line by clicking three points.
  - Mode automatically turns off after creation.

---

## Summary of Controls
| Key | Action |
|-----|--------|
| **N** | Start new lane group |
| **S** | Save selection / proceed |
| **Q** | Quit adjustment mode |

---
