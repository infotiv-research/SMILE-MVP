

# How to populate the `Data` and `models` directories

- Volvo data: - WP6 - PrototypesDemonstrators (Volvo)\datasets\(CONFIDENTIAL) Tuve dataset

- MVP codes: WP6 - PrototypesDemonstrators (Volvo)\datasets\(CONFIDENTIAL) Tuve dataset\MVP

- Anomaly detections (Jesper): WP6 - PrototypesDemonstrators (Volvo)\datasets\ood_detections

- Play JSON-data/final.json and display trajectory + maneuver classes (Thanh). Original codes are at P119522 - SMILE IV - General\Volvo Trajectories. python .\trajectory_player.py --gui

- Play Joakim predictions (load temp4_with_analysis.json). Original codes are at P119522 - SMILE IV - General\Joakim:
play_lstm_results.py

- Run MVP simple (still need old data file thanh_object_tracks2) DOES NOT WORK YET :(: 
tuve_mvp.py

# Quick start

Run `tree -L 2` to make sure that all the data files are correctly placed in your `Data` and `Models` directory:

```
.
├── Data
│   ├── (CONFIDENTIAL) Tuve dataset
│   └── obstacles_and_forklifts
├── JSON-data
│   ├── analysis.json
│   ├── final.json
│   ├── object_tracks.json
│   └── predictions.json
├── Models
│   ├── camera_visibility_lookup_table.pkl
│   └── model_20260416_131406.npz
├── MVP.ipynb
├── play_lstm_results.py
├── README.md
├── requirements.txt
├── trajectory_player.py
├── trajectory_visualizer.py
└── tuve_mvp.py
```

Now create a new virtual environments to install dependencies:

```
python -m venv smile-env
% Activate it on Windows:  smile-env\Scripts\activate
source smile-env/bin/activate
pip install -r requirements.txt
```
Run the project:

```
python trajectory_player.py  --gui
```
```
python -m ipykernel install --user --name=smile-env --display-name "Python (SMILE-IV)"

jupyter notebook
```
# Contribute

Before saving and committing your jupyter notebooks, go to the Jupyter menu:
> Kernel -> Restart & Clear All Output.

or initialize nbstripout in your repo to let git doing that automatically
```
nbstripout --install
```

# Issues
if you get the following error message:
```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
```
install:

```
sudo apt install libxcb-cursor0
```