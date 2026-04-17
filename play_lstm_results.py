import json
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QTextEdit, QGraphicsTextItem
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPen, QBrush, QColor


class ReplayViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trajectory Replay & Analysis Viewer")
        self.resize(1600, 1000)

        self.frames = []
        self.analysis = {}
        self.track_frame_map = {}  # track_id → list of global frame indices
        self.history = {}          # track_id → list of (x, y)

        self.current_frame_idx = 0

        self.build_ui()

    # ---------------------------------------------------------
    # UI SETUP
    # ---------------------------------------------------------
    def build_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Graphics canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view, 3)

        # Right panel
        right = QVBoxLayout()
        layout.addLayout(right, 1)

        # Load JSON
        btn_load = QPushButton("Load JSON")
        btn_load.clicked.connect(self.load_json)
        right.addWidget(btn_load)

        # Frame slider
        right.addWidget(QLabel("Frame"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.update_frame_index)
        right.addWidget(self.slider)

        # Analysis panel
        right.addWidget(QLabel("Analysis (all objects in current frame):"))
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        right.addWidget(self.analysis_text, 1)

    # ---------------------------------------------------------
    # LOAD JSON
    # ---------------------------------------------------------
    def load_json(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON (*.json)")
        if not file:
            return

        with open(file, "r") as f:
            root = json.load(f)

        self.frames = root["data"]
        self.analysis = root["analysis"]
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.frames) - 1)

        # Build per-track global frame index map
        self.track_frame_map = {}
        for frame_idx, frame in enumerate(self.frames):
            for obj in frame["object_list"]:
                track_id = obj[0]
                self.track_frame_map.setdefault(track_id, []).append(frame_idx)

        # Reset histories
        self.history = {}

        self.update_scene()

    # ---------------------------------------------------------
    # SLIDER MOVED
    # ---------------------------------------------------------
    def update_frame_index(self, idx):
        self.current_frame_idx = idx
        self.update_scene()

    # ---------------------------------------------------------
    # FIND ACTIVE WINDOW (CORRECT GLOBAL INDEX LOGIC)
    # ---------------------------------------------------------
    def find_active_window(self, traj, frame_idx):
        track_id = traj["track_id"]
        global_index_list = self.track_frame_map.get(track_id, [])
        if not global_index_list:
            return None

        for w in traj["windows"]:
            local_start = w["start_idx"]
            local_end = w["end_idx"]

            # Skip if window indices exceed trajectory length
            if local_start >= len(global_index_list) or local_end >= len(global_index_list):
                continue

            global_start = global_index_list[local_start]
            global_end = global_index_list[local_end]

            if global_start <= frame_idx <= global_end:
                return w

        return None

    # ---------------------------------------------------------
    # RENDER EVERYTHING
    # ---------------------------------------------------------
    def update_scene(self):
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]
        objects = frame["object_list"]

        # Clear scene
        self.scene.clear()

        # Keep track of analysis text
        analysis_lines = []

        # Add header with frame/time
        timestamp_ms = frame["time_stamp"]
        timestamp_s = timestamp_ms / 1000.0
        total_frames = len(self.frames)

        header = (
            f"Frame {self.current_frame_idx+1}/{total_frames}\n"
            f"Timestamp: {timestamp_s:.3f} sec\n\n"
        )

        # Update trajectories and draw them
        for obj in objects:
            track_id, x, y, _ = obj

            # Remember history
            self.history.setdefault(track_id, []).append((x, y))

        # Draw trajectories
        pen_traj = QPen(QColor("orange"))
        pen_traj.setWidthF(0.08)

        for track_id, pts in self.history.items():
            for i in range(1, len(pts)):
                x1, y1 = pts[i - 1]
                x2, y2 = pts[i]
                self.scene.addLine(x1, -y1, x2, -y2, pen_traj)

        # Draw objects (after trajectories)
        for obj in objects:
            track_id, x, y, _ = obj

            # Draw object as a circle
            r = 0.5
            circle = QGraphicsEllipseItem(QRectF(x - r, -y - r, 2*r, 2*r))
            circle.setBrush(QBrush(QColor("dodgerblue")))
            circle.setPen(QPen(Qt.GlobalColor.black))
            self.scene.addItem(circle)

            # Draw object ID label
            label = QGraphicsTextItem(str(track_id))
            label.setDefaultTextColor(QColor("white"))
            label.setPos(x + 0.35, -y - 0.35)
            self.scene.addItem(label)

        # Build analysis for all objects in this frame
        for obj in objects:
            track_id = obj[0]

            traj = next((t for t in self.analysis["trajectories"]
                         if t["track_id"] == track_id), None)

            analysis_lines.append(f"Track {track_id}")

            if traj is None:
                analysis_lines.append("  No analysis available.\n")
                continue

            analysis_lines.append(f"  Start ROI: {traj['start_roi']}")
            analysis_lines.append(f"  End ROI:   {traj['end_roi']}")

            # Choose correct window
            window = self.find_active_window(traj, self.current_frame_idx)
            if window is None:
                analysis_lines.append("  No window covers this frame.\n")
                continue

            # Activity
            act = window["activity"]
            
            p_still = act["p_still"] if act["p_still"] is not None else 0.0
            p_loading = act["p_loading"] if act["p_loading"] is not None else 0.0
            p_transport = act["p_transport"] if act["p_transport"] is not None else 0.0

            analysis_lines.append(f"  Predicted route: {act['pred_cls']}")
            analysis_lines.append(f"  Still: {p_still:.3f}")
            analysis_lines.append(f"  Loading: {p_loading:.3f}")
            analysis_lines.append(f"  Transport: {p_transport:.3f}")


            # Route probabilities
            analysis_lines.append("  Route probabilities:")
            for r in window["routes"]:
                analysis_lines.append(
                    f"    {r['route_type']}: {r['prob']:.3f}"
                )

            analysis_lines.append("")

        # Write to panel
        self.analysis_text.setPlainText(header + "\n".join(analysis_lines))


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ReplayViewer()
    viewer.show()
    sys.exit(app.exec())