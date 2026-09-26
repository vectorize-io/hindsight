"""Validate shared dependencies and emit NVIDIA requirements during image build."""

from importlib.metadata import distributions, version
from pathlib import Path

from packaging.requirements import Requirement

runtime = next(d for d in distributions(path=["/opt/hindsight-onnx-gpu"]) if d.metadata["Name"] == "onnxruntime-gpu")
nvidia = []
for entry in runtime.requires or []:
    requirement = Requirement(entry)
    if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in ("", "cuda", "cudnn")):
        continue
    if requirement.name.startswith("nvidia-"):
        nvidia.append(str(requirement).split(";")[0])
    else:
        assert requirement.specifier.contains(version(requirement.name)), f"Base dependency incompatible: {requirement}"
Path("/tmp/onnx-nvidia.txt").write_text("\n".join(nvidia))
