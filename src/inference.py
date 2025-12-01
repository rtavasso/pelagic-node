"""
Inference engine for marine acoustic classifier.

Loads quantized ONNX model and runs inference on mel-spectrograms.
Per spec: Input [1, 3, 224, 224], Output softmax [p_bg, p_vessel, p_cetacean].
"""

import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, List

from config import MODEL_PATH, CLASS_NAMES


class InferenceEngine:
    """
    ONNX Runtime inference wrapper.

    Per spec Section C:
    - Model: MobileNetV2 (Quantized ONNX)
    - Input: 224x224 Mel-Spectrogram (Stacked 3-channel)
    - Output: Softmax probability vector [p_background, p_vessel, p_cetacean]
    - Confidence: max(output) — the highest probability value
    - Predicted Class: argmax(output) — index of highest probability

    Note: ImageFolder sorts classes alphabetically, so actual order is:
    [background, cetacean, vessel] -> indices [0, 1, 2]
    We remap to match spec: Background=0, Vessel=1, Cetacean=2
    """

    # Mapping from ImageFolder order to spec order
    # ImageFolder: background=0, cetacean=1, vessel=2
    # Spec order:  Background=0, Vessel=1, Cetacean=2
    IMAGEFOLDER_TO_SPEC = {
        0: 0,  # background -> Background
        1: 2,  # cetacean -> Cetacean
        2: 1,  # vessel -> Vessel
    }

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize ONNX Runtime session.

        Args:
            model_path: Path to ONNX model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model into inference session."""
        try:
            # Use CPU execution provider (simulating edge device)
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            # Verify input/output shapes
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]

            print(f"Model loaded: {self.model_path}")
            print(f"  Input: {input_info.name}, shape {input_info.shape}")
            print(f"  Output: {output_info.name}, shape {output_info.shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def run(self, spectrogram: np.ndarray) -> Tuple[int, float, List[float]]:
        """
        Run inference on a spectrogram.

        Args:
            spectrogram: np.ndarray of shape [1, 3, 224, 224], float32, [0, 1]

        Returns:
            Tuple of (class_id, confidence, probabilities):
            - class_id: 0=Background, 1=Vessel, 2=Cetacean (spec order)
            - confidence: max probability (0.0 to 1.0)
            - probabilities: [p_background, p_vessel, p_cetacean] in spec order
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")

        # Validate input shape
        expected_shape = (1, 3, 224, 224)
        if spectrogram.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {spectrogram.shape}")

        # Ensure float32
        spectrogram = spectrogram.astype(np.float32)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: spectrogram})

        # Get logits and apply softmax
        logits = outputs[0][0]  # Shape: [3]
        probs_imagefolder = self._softmax(logits)

        # Get prediction in ImageFolder order
        imagefolder_class = int(np.argmax(probs_imagefolder))
        confidence = float(np.max(probs_imagefolder))

        # Remap class to spec order
        spec_class = self.IMAGEFOLDER_TO_SPEC[imagefolder_class]

        # Remap probabilities to spec order: [p_bg, p_vessel, p_cetacean]
        # ImageFolder order: [background=0, cetacean=1, vessel=2]
        # Spec order:        [Background=0, Vessel=1, Cetacean=2]
        probs_spec = [
            float(probs_imagefolder[0]),  # background -> Background
            float(probs_imagefolder[2]),  # vessel -> Vessel
            float(probs_imagefolder[1]),  # cetacean -> Cetacean
        ]

        return spec_class, confidence, probs_spec

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()


def load_model(model_path: str = MODEL_PATH) -> InferenceEngine:
    """
    Factory function to load inference engine.

    Args:
        model_path: Path to ONNX model

    Returns:
        Initialized InferenceEngine instance
    """
    return InferenceEngine(model_path)


def run_model(engine: InferenceEngine, spectrogram: np.ndarray) -> Tuple[int, float, List[float]]:
    """
    Run inference using provided engine.

    Args:
        engine: Loaded InferenceEngine instance
        spectrogram: Input spectrogram array

    Returns:
        Tuple of (class_id, confidence, probabilities)
    """
    return engine.run(spectrogram)
