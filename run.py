import argparse
import time

import cv2

from ledetector.detector.model import ObjectDetector
from ledetector.detector.stream import VideoStream
from ledetector.leds.matrix import LEDMatrix

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modeldir", help="Folder the .tflite file is located in", required=True
)
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator to speed up detection",
    action="store_true",
)
parser.add_argument(
    "--n_leds",
    help="Number of leds",
    default=100,
    type=int
)
parser.add_argument(
    "--sleep",
    help="Time in seconds to sleep between detections",
    default=0,
    type=float
)

args = parser.parse_args()

od = ObjectDetector(model_dir=args.modeldir)

imW = 640
imH = 480

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(640, 480), framerate=30).start()
time.sleep(1)

matrix = LEDMatrix(args.n_leds)

while True:
    # Grab frame from video stream
    frame1 = videostream.read()
    frame = frame1.copy()

    # Acquire frame and resize to expected shape [1xHxWx3]
    input_data = od.process_input_image(frame1)
    boxes, classes, scores = od.get_filtered_results(input_data)
    x = None
    if len(boxes) > 0:
        x = (boxes[0][1] + boxes[0][3]) / 2.0
        print(f"X: {x:.3f}", end="\r")
    matrix.display_detection(x)
    time.sleep(args.sleep)
