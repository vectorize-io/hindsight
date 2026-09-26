"""Build-time import checks; device activation still requires a real GPU run."""

import importlib.util
from importlib.metadata import distributions, version
from pathlib import Path

import flatbuffers
import google.protobuf
import numpy
import onnxruntime as ort
import packaging
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.embeddings import OnnxEmbeddings
from magika import Magika

assert ort.__file__.startswith("/opt/hindsight-onnx-gpu/")
assert "CUDAExecutionProvider" in ort.get_available_providers()
assert importlib.util.find_spec("torch") is None
for distribution in distributions(path=["/opt/hindsight-onnx-gpu"]):
    name = distribution.metadata["Name"]
    assert name == "onnxruntime-gpu" or name.startswith("nvidia-"), name
for module in (numpy, google.protobuf, packaging, flatbuffers):
    assert Path(module.__file__).is_relative_to("/app/api/.venv"), module.__file__
    print(module.__name__, module.__file__)
print({name: version(name) for name in ("numpy", "protobuf", "packaging", "flatbuffers")})
assert HindsightConfig.from_env().embeddings_onnx_device == "cpu"
assert OnnxEmbeddings is not None
Magika()
print(ort.__version__, ort.get_available_providers())
