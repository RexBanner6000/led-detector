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
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help="Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.",
    default="1280x720",
)
parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator to speed up detection",
    action="store_true",
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

matrix = LEDMatrix(3)

while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    frame = frame1.copy()

    # Acquire frame and resize to expected shape [1xHxWx3]
    input_data = od.process_input_image(frame1)
    boxes, classes, scores = od.get_filtered_results(input_data)

    matrix.flash_green(len(boxes) > 0)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > od.threshold) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = od.labels[
                int(classes[i])
            ]  # Look up object name from "labels" array using class index
            label = "%s: %d%%" % (
                object_name,
                int(scores[i] * 100),
            )  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            label_ymin = max(
                ymin, labelSize[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                frame,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                frame,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )  # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(
        frame,
        "FPS: {0:.2f}".format(frame_rate_calc),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow("Object detector", frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
