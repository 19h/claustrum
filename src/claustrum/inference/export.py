"""Model export and optimization for production deployment.

Implements optimization strategies from the plan:
- ONNX export for portable inference
- INT8 quantization for 4x memory reduction
- TensorRT compilation for NVIDIA GPUs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, int] = (1, 512),
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> None:
    """Export model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path for ONNX file
        input_shape: (batch_size, seq_length) for dummy input
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch/sequence dimensions
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy inputs
    device = next(model.parameters()).device
    dummy_input_ids = torch.randint(0, 1000, input_shape, device=device)
    dummy_attention_mask = torch.ones(input_shape, device=device)

    # Define dynamic axes
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"},
        }

    # Export
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes_config,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Exported model to {output_path}")

    # Verify export
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except ImportError:
        print("onnx package not installed, skipping verification")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


def quantize_model(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    quantization_type: str = "dynamic",  # "dynamic", "static"
    calibration_data: Optional[list] = None,
) -> None:
    """Quantize ONNX model for reduced size and faster inference.

    INT8 quantization provides ~4x memory reduction with 1-3% accuracy loss.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path for quantized model
        quantization_type: "dynamic" or "static"
        calibration_data: Data for static quantization calibration
    """
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)

    if quantization_type == "dynamic":
        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
    elif quantization_type == "static":
        if calibration_data is None:
            raise ValueError("calibration_data required for static quantization")

        # Create calibration data reader
        class CalibrationDataReader:
            def __init__(self, data):
                self.data = data
                self.index = 0

            def get_next(self):
                if self.index >= len(self.data):
                    return None
                item = self.data[self.index]
                self.index += 1
                return item

        quantize_static(
            str(onnx_path),
            str(output_path),
            CalibrationDataReader(calibration_data),
            weight_type=QuantType.QInt8,
        )
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    print(f"Quantized model saved to {output_path}")


class ONNXInferenceSession:
    """ONNX Runtime inference session wrapper.

    Provides optimized inference with automatic hardware acceleration.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        use_gpu: bool = True,
    ):
        import onnxruntime as ort

        self.model_path = Path(model_path)

        # Configure providers
        providers = []
        if use_gpu:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "TensorrtExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session
        self.session = ort.InferenceSession(
            str(model_path),
            session_options,
            providers=providers,
        )

        # Get input/output names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"Loaded ONNX model from {model_path}")
        print(f"Providers: {self.session.get_providers()}")

    def __call__(
        self,
        input_ids,
        attention_mask,
    ):
        """Run inference.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask

        Returns:
            Dictionary with model outputs
        """
        import numpy as np

        # Convert to numpy if needed
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        if hasattr(attention_mask, "numpy"):
            attention_mask = attention_mask.numpy()

        # Ensure correct dtype
        input_ids = input_ids.astype(np.int64)
        attention_mask = attention_mask.astype(np.int64)

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        return {name: out for name, out in zip(self.output_names, outputs)}

    def embed(
        self,
        input_ids,
        attention_mask,
    ):
        """Get embeddings (convenience method).

        Returns:
            Embedding array
        """
        outputs = self(input_ids, attention_mask)
        return outputs.get("embeddings", outputs[self.output_names[0]])


def optimize_for_tensorrt(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    max_batch_size: int = 32,
    fp16: bool = True,
) -> None:
    """Optimize ONNX model with TensorRT.

    Provides significant speedup on NVIDIA GPUs.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path for TensorRT engine
        max_batch_size: Maximum batch size for optimization
        fp16: Use FP16 precision
    """
    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with trt.Builder(TRT_LOGGER) as builder:
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

            with builder.create_network(network_flags) as network:
                with trt.OnnxParser(network, TRT_LOGGER) as parser:
                    # Read ONNX model
                    with open(onnx_path, "rb") as f:
                        if not parser.parse(f.read()):
                            for i in range(parser.num_errors):
                                print(parser.get_error(i))
                            raise RuntimeError("Failed to parse ONNX model")

                    # Configure builder
                    config = builder.create_builder_config()
                    config.max_workspace_size = 1 << 30  # 1GB

                    if fp16 and builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)

                    # Set dynamic shapes
                    profile = builder.create_optimization_profile()
                    profile.set_shape(
                        "input_ids",
                        (1, 1),  # min
                        (max_batch_size // 2, 256),  # opt
                        (max_batch_size, 512),  # max
                    )
                    profile.set_shape(
                        "attention_mask",
                        (1, 1),
                        (max_batch_size // 2, 256),
                        (max_batch_size, 512),
                    )
                    config.add_optimization_profile(profile)

                    # Build engine
                    engine = builder.build_serialized_network(network, config)

                    if engine is None:
                        raise RuntimeError("Failed to build TensorRT engine")

                    # Save engine
                    with open(output_path, "wb") as f:
                        f.write(engine)

                    print(f"TensorRT engine saved to {output_path}")

    except ImportError:
        print("TensorRT not available. Install with: pip install tensorrt")
    except Exception as e:
        print(f"TensorRT optimization failed: {e}")
