from ultralytics import YOLO


class VehicleTracker:
    def __init__(self) -> None:
        self.model = YOLO('yolov8n.pt')
        self.model.fuse()

    def track(self,frame):
        results = self.model.track(frame,persist=True,classes= 2)
        annotated_frame = results[0].plot()
        return annotated_frame