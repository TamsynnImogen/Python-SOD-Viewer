# Python-SOD-Viewer
## SOD Viewer — Install & Run
### Requirements

- Python: 3.9+ (tested on 3.10–3.13)

- GPU/Driver: OpenGL 3.3 or newer

- OS: Windows / Linux / macOS

- Packages: pyglet >= 2.0 (only dependency)

- Tk file dialogs: tkinter (bundled with Python on Windows/macOS; on Linux install the OS package)

#### Linux tkinter packages:

- Debian/Ubuntu: sudo apt install python3-tk

- Fedora: sudo dnf install python3-tkinter

- Arch: sudo pacman -S tk

### Quick install
#### (optional but recommended) create a venv
python -m venv .venv
#### activate
##### Windows:
.venv\Scripts\activate
##### macOS/Linux:
source .venv/bin/activate

### install dependency
python -m pip install --upgrade pip
pip install "pyglet>=2.0,<3"


##### Verify:

python -c "import pyglet; print('pyglet', pyglet.version)"

### Run
###### open the app (no file preloaded)
python sod_viewer.py

###### or start directly with a model and base folder
python sod_viewer.py "path/to/model.sod" --tex-root "path/to/base"

### Texture lookup

The viewer searches (case-insensitive) under:

<base>/sod
<base>/textures
<base>/textures/rgb


Formats: .tga and .dds (case-insensitive).

### Controls

- O: Open .SOD

- L: Pick base folder (parent of sod / textures)

- S: Save as SOD (choose 1.8 / 1.93)

- A: Play/Pause transform animations

- B: Toggle Borg branch (node “borg” or names containing “borg” + descendants)

- M: Toggle damage meshes (heuristic: names/materials/textures containing “damage”/“dmg”)

- W: Wireframe

- T: Toggle textures

- C: Toggle back-face culling

- F: Flip front-face winding (CCW/CW)

- D: Dump texture resolution attempts to console

- Space: Reset view

- Left-drag: Orbit camera

- Scroll: Zoom

- Esc: Quit

### Troubleshooting

- “OpenGL version” / blank window: Update GPU drivers; ensure OpenGL 3.3+ is available.

- _tkinter not found (Linux): Install python3-tk (see above).

- Textures not found: Use L to set the correct base. Press D to see where it looked.

- Wireframe or face flip hides the UI: The viewer forces a separate 2D UI pass; if text still looks odd, ensure you’re on pyglet 2.x.
