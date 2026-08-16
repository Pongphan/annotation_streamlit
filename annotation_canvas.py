import base64
import copy
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


COMPONENT_DIR = Path(__file__).parent / "annotation_canvas_component"
EMPTY_DRAWING = {"version": "meinlab-annotation-canvas-1.1", "objects": []}

_annotation_canvas = components.declare_component(
    "meinlab_annotation_canvas",
    path=str(COMPONENT_DIR),
)


@dataclass
class CanvasResult:
    image_data: Any = None
    json_data: Optional[Dict[str, Any]] = None


def _image_to_data_url(image: Image.Image) -> Tuple[str, str]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    digest = hashlib.md5(png_bytes).hexdigest()
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}", digest


def _normalize_drawing(drawing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(drawing, dict):
        return dict(EMPTY_DRAWING)

    objects = drawing.get("objects", [])

    if not isinstance(objects, list):
        objects = []

    normalized_objects = []

    for index, raw_object in enumerate(objects):
        if not isinstance(raw_object, dict):
            continue

        obj = copy.deepcopy(raw_object)

        if not obj.get("_annotationId"):
            serialized = json.dumps(obj, sort_keys=True, default=str)
            digest = hashlib.md5(
                f"{index}:{serialized}".encode("utf-8")
            ).hexdigest()
            obj["_annotationId"] = f"annotation-{digest}"

        normalized_objects.append(obj)

    return {
        "version": drawing.get("version", EMPTY_DRAWING["version"]),
        "objects": normalized_objects,
        "background": drawing.get("background", ""),
    }


def st_canvas(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Optional[Image.Image] = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: Optional[Dict[str, Any]] = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key: Optional[str] = None,
) -> CanvasResult:
    width = int(width)
    height = int(height)
    component_key = key or "meinlab_annotation_canvas"
    state_key = f"{component_key}_drawing_json"
    config_key = f"{component_key}_drawing_config"

    background_image_url = None
    background_digest = "no-background"

    if background_image is not None:
        background = background_image.copy()
        target_size = (width, height)

        if background.size != target_size:
            background = background.resize(target_size)

        background_image_url, background_digest = _image_to_data_url(background)
        background_color = ""

    canvas_config = f"{background_digest}:{width}:{height}"

    if st.session_state.get(config_key) != canvas_config:
        st.session_state[config_key] = canvas_config
        st.session_state[state_key] = _normalize_drawing(initial_drawing)

    drawing = _normalize_drawing(
        initial_drawing or st.session_state.get(state_key)
    )

    component_value = _annotation_canvas(
        fillColor=fill_color,
        strokeWidth=int(stroke_width),
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=drawing,
        displayToolbar=display_toolbar,
        pointDisplayRadius=point_display_radius,
        updateStreamlit=update_streamlit,
        key=component_key,
        default=drawing,
    )

    json_data = _normalize_drawing(component_value)

    if json_data != st.session_state.get(state_key):
        st.session_state[state_key] = json_data
        st.rerun()

    return CanvasResult(json_data=json_data)
