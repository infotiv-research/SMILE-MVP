import sys
import json
import math
from collections import defaultdict

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QSlider, QStyle, QMainWindow, QGraphicsScene, QGraphicsView,
    QSpinBox, QGraphicsTextItem
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QPolygonF, QPen, QColor, QPixmap, 
    QTransform
)


class TrackingViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking Viewer with Rotation + Velocity Arrows")

        self.frames = []
        self.current_frame_idx = 0
        self.background_pixmap = None

        self.track_paths = defaultdict(list)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Top bar UI
        load_json_btn = QPushButton("Load Tracking JSON")
        load_json_btn.clicked.connect(self.load_json)

        load_img_btn = QPushButton("Load Background Image")
        load_img_btn.clicked.connect(self.load_background_image)

        self.play_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "")
        self.play_btn.clicked.connect(self.toggle_playback)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.update_fps)

        self.timestamp_label = QLabel("Timestamp: -")

        top_bar = QHBoxLayout()
        top_bar.addWidget(load_json_btn)
        top_bar.addWidget(load_img_btn)
        top_bar.addWidget(self.play_btn)
        top_bar.addWidget(QLabel("FPS:"))
        top_bar.addWidget(self.fps_spin)
        top_bar.addWidget(self.timestamp_label)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for playback
        self.timer = QTimer()
        self.update_fps()
        self.timer.timeout.connect(self.next_frame)

        self.global_bounds = QRectF()
        self.scene_set = False

    # ---------------------------------------------------------
    def load_json(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Tracking JSON", "", "JSON Files (*.json)")
        if not file:
            return

        with open(file, "r") as f:
            self.frames = json.load(f)

        self.slider.setMaximum(len(self.frames) - 1)
        self.track_paths.clear()
        self.current_frame_idx = 0
        self.compute_global_bounds()
        self.draw_frame()

    # ---------------------------------------------------------
    def load_background_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Background Image", "", "Images (*.png *.jpg *.bmp)")
        if not file:
            return
        self.background_pixmap = QPixmap(file)
        self.draw_frame()

    # ---------------------------------------------------------
    def update_fps(self):
        fps = self.fps_spin.value()
        self.timer.setInterval(int(1000 / fps))

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.timer.start()
            self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def next_frame(self):
        if not self.frames:
            return
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        self.slider.setValue(self.current_frame_idx)
        self.draw_frame()

    def slider_changed(self, value):
        self.current_frame_idx = value
        self.draw_frame()

    # ---------------------------------------------------------
    def compute_global_bounds(self):
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for frame in self.frames:
            for obj in frame["object_list"]:
                x = obj["position_3d"]["x"]
                y = obj["position_3d"]["y"]
                min_x = min(min_x, x); max_x = max(max_x, x)
                min_y = min(min_y, y); max_y = max(max_y, y)

                for px, py in obj.get("associated_polygon", []):
                    min_x = min(min_x, px); max_x = max(max_x, px)
                    min_y = min(min_y, py); max_y = max(max_y, py)

        margin = 100
        self.global_bounds = QRectF(min_x - margin, min_y - margin,
                                    (max_x - min_x) + 2 * margin,
                                    (max_y - min_y) + 2 * margin)

        self.scene_set = False

    # ---------------------------------------------------------
    def draw_frame(self):
        if not self.frames:
            return

        self.scene.clear()
        frame = self.frames[self.current_frame_idx]
        self.timestamp_label.setText(f"Timestamp: {frame.get('time_stamp', '-')}")

        if not self.scene_set:
            self.scene.setSceneRect(self.global_bounds)
            self.view.fitInView(self.global_bounds, Qt.AspectRatioMode.KeepAspectRatio)
            self.scene_set = True

        # Background
        if self.background_pixmap:
            bg = self.scene.addPixmap(self.background_pixmap)
            bg.setZValue(-1000)

        # Objects
        for obj in frame["object_list"]:
            self.draw_object(obj)

        # Trajectory lines
        self.draw_trajectories()

    # ---------------------------------------------------------
    def draw_object(self, obj):
        x = obj["position_3d"]["x"]
        y = obj["position_3d"]["y"]
        track_id = obj["track_id"]

        # store path
        self.track_paths[track_id].append((x, y))

        # polygon
        pts = obj.get("associated_polygon", [])
        if pts:
            poly = self.scene.addPolygon(
                QPolygonF([QPointF(px, py) for px, py in pts]),
                QPen(QColor(255, 0, 0), 2)
            )

        # Determine rotation angle
        heading = obj.get("heading", {})
        rad = heading.get("rad")
        deg = heading.get("deg")

        if rad is not None:
            angle = rad
        elif deg is not None:
            angle = math.radians(deg)
        else:
            # fallback: compute from velocity
            vx = obj["velocity_3d"]["x"]
            vy = obj["velocity_3d"]["y"]
            if abs(vx) > 1e-3 or abs(vy) > 1e-3:
                angle = math.atan2(vy, vx)
            else:
                angle = 0

        # Rotatable bbox
        bbox = obj.get("bbox_moving_avg", {})
        w = bbox.get("width", 30)
        h = bbox.get("length", 15)

        cx, cy = x, y

        # rectangle initially centered at origin
        rect = self.scene.addRect(-w/2, -h/2, w, h, QPen(QColor(0, 255, 0), 2))
        rect.setPos(cx, cy)

        # Apply rotation
        transform = QTransform()
        transform.rotate(math.degrees(angle))
        rect.setTransform(transform, True)

        # Draw center
        self.scene.addEllipse(cx - 2, cy - 2, 4, 4, QPen(QColor(0, 0, 255), 2))

        # Label (track ID)
        label = QGraphicsTextItem(str(track_id))
        label.setDefaultTextColor(QColor(255, 255, 0))
        label.setPos(cx + 5, cy - 20)
        self.scene.addItem(label)

        # Velocity arrow
        self.draw_velocity_arrow(cx, cy, obj)

    # ---------------------------------------------------------
    def draw_velocity_arrow(self, x, y, obj):
        vx = obj["velocity_3d"]["x"]
        vy = obj["velocity_3d"]["y"]

        speed = math.hypot(vx, vy)
        if speed < 1e-3:
            return

        scale = 10  # tuning factor for visibility
        x2 = x + vx * scale
        y2 = y + vy * scale

        pen = QPen(QColor(255, 128, 0), 3)
        self.scene.addLine(x, y, x2, y2, pen)

        # arrowhead
        angle = math.atan2(vy, vx)
        arrow_size = 6
        left = QPointF(
            x2 - arrow_size * math.cos(angle - math.pi / 6),
            y2 - arrow_size * math.sin(angle - math.pi / 6)
        )
        right = QPointF(
            x2 - arrow_size * math.cos(angle + math.pi / 6),
            y2 - arrow_size * math.sin(angle + math.pi / 6)
        )

        self.scene.addLine(x2, y2, left.x(), left.y(), pen)
        self.scene.addLine(x2, y2, right.x(), right.y(), pen)

    # ---------------------------------------------------------
    def draw_trajectories(self):
        pen = QPen(QColor(0, 200, 255), 2)

        for tid, pts in self.track_paths.items():
            for i in range(1, len(pts)):
                x1, y1 = pts[i - 1]
                x2, y2 = pts[i]
                self.scene.addLine(x1, y1, x2, y2, pen)


# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TrackingViewer()
    viewer.resize(1400, 900)
    viewer.show()
    sys.exit(app.exec())