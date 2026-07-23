"""The pet-monitoring pack's event vocabulary.

Every sensor emits into the one :class:`~convilyn_edge.envelope.EventEnvelope` shell (data-flow
consistency); these constants name the pack's ``event_type`` values and its payload schema. The
scenario nouns (camera / feeder / water / litter) live here in the pack, never in the substrate.
"""

from __future__ import annotations

#: The pack's payload schema tag (carried on ``EventEnvelope.event_schema``).
SENSOR_SCHEMA = "pet.sensor.v1"

#: A camera frame became available (payload: ``frame_id``, ``motion``, optional ``image_ref``).
EVENT_CAMERA_FRAME = "pet.camera.frame"
#: A feeder reading (payload: ``level_pct`` 0-100, ``jammed`` bool).
EVENT_FEEDER_READING = "pet.feeder.reading"
#: A water reading (payload: ``level_ml``).
EVENT_WATER_READING = "pet.water.reading"
#: A litter reading (payload: ``uses`` since last clean).
EVENT_LITTER_READING = "pet.litter.reading"

__all__ = [
    "SENSOR_SCHEMA",
    "EVENT_CAMERA_FRAME",
    "EVENT_FEEDER_READING",
    "EVENT_WATER_READING",
    "EVENT_LITTER_READING",
]
