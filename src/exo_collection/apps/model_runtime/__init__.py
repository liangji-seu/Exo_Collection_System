"""Online model testing runtime (在线测试).

Real-time per-modality subscriptions, a pluggable model runtime, and a Mock
torque output.  See ``spec.ModelSpec`` for the JSON contract that ties the three
together, and ``main.main`` for the GUI entry point.
"""

from exo_collection.apps.model_runtime.spec import ModelSpec, load_model_spec

__all__ = ["ModelSpec", "load_model_spec"]
