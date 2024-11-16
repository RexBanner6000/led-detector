import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class ObjectDetector:
    def __init__(
        self,
        model_dir: str,
        graph_name: str = "detect.tflite",
        label_map_name: str = "labelmap.txt",
        threshold: float = 0.5,
        resolution: Tuple[int] = (640, 480),
    ) -> None:
        self.import_libraries()
        self.model_dir = Path(model_dir)
        self.graph_path = self.model_dir / graph_name
        self.labels_path = self.model_dir / label_map_name
        self.interpreter = Interpreter(model_path=self.graph_path)
        self.threshold = threshold
        self.img_width = resolution[0]
        self.img_height = resolution[1]
        self.import_libraries()
        self.labels = self.get_labels()

        self.height = None
        self.width = None
        self.floating_model = None
        self.outname = None
        self.get_model_details()

        self.input_mean = 127.5
        self.input_std = 127.5

        self.indices = self.get_indices()

        self.interpreter.allocate_tensors()

    @staticmethod
    def import_libraries() -> None:
        pkg = importlib.util.find_spec("tflite_runtime")
        if pkg:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter

    def get_labels(self) -> List[str]:
        with open(self.labels_path, "r") as fp:
            labels = [line.strip() for line in fp.readlines()]

        if labels[0] == "???":
            del labels[0]
        return labels

    def get_model_details(self) -> None:
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.height = input_details[0]["shape"][1]
        self.width = input_details[0]["shape"][2]
        self.floating_model = input_details[0]["dtype"] == np.float32
        self.outname = output_details[0]["name"]

    def get_indices(self) -> Dict[str, int]:
        if "StatefulPartitionedCall" in self.outname:  # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else:  # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2

        return {
            "boxes_idx": boxes_idx,
            "classes_idx": classes_idx,
            "scores_idx": scores_idx,
        }


if __name__ == "__main__":
    od = ObjectDetector(model_dir="./TFLite_model")
