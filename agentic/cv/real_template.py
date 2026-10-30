"""TEMPLATE for the real CV model. Copy to e.g. ``agentic/cv/real_backend.py``, fill it in, then set
``RG_CV_BACKEND=real`` and ``RG_CV_REAL_MODULE=agentic.cv.real_backend``. Nothing else needs to change.

Minimum: a ``DefectDetector``. Every other tool is optional; missing ones are reported as "skipped".

Example detector for a Roboflow-trained defect-clothes model::

    from agentic.cv.label_map import roboflow_to_detections

    class RoboflowDetector:
        def detect(self, image: ImageInput) -> list[Detection]:
            w, h = ...  # image size in pixels
            predictions = self._model.infer(image.path)   # list of dicts: class, confidence, x, y, width, height
            return roboflow_to_detections(predictions, w, h).detections

Contract reminders (see agentic/cv/interfaces.py): normalised 0-1 boxes, "no defect" = empty list,
raise on real failures (they are recorded and the case degrades safely), echo the same ``image_id``.
"""
from agentic.cv.interfaces import CVToolbox


def build_toolbox() -> CVToolbox:
    raise NotImplementedError("fill in agentic/cv/real_template.py (or point RG_CV_REAL_MODULE at your module)")
