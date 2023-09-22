# Yolov8 Region Createable and Moveable Count Objects

- Create Regions

## Hotkeys
~~~~~~~
+--------------------+----------------------+
| s + enter   | Create polygon boxs         |
+--------------------+----------------------+
| m           | Move a polygon box          |
+--------------------+----------------------+
| q           | close the video             |
+--------------------+----------------------+

~~~~~~~~~~~~~~~~~~~~~~~~~

## How to Run
~~~~
# If you want to save results
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img

# If you want to run model on CPU
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# If you want to change model file
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/model.pt"

# If you dont want to save results
python yolov8_region_counter.py --source "path/to/video.mp4" --view-img

~~~~~~~~~~~~~~~~~~~~~
## Usage Options

- `--source:` Specifies the path to the video file you want to run inference on.
- `--device:` Specifies the device cpu or 0
- `--save-img:` Flag to save the detection results as images.
- `--weights:` Specifies a different YOLOv8 model file (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt).
- `--line-thickness:` Specifies the bounding box thickness
- `--region-thickness:` Specifies the region boxes thickness
- `--track-thickness:` Specifies the track line thickness

## Where Can I Access Additional Information?

https://github.com/RizwanMunawar/ultralytics/tree/main/examples/YOLOv8-Region-Counter
