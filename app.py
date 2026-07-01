import io
import json
import zipfile
import hashlib
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from annotation_canvas import st_canvas

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Meinlab Image Annotator",
    page_icon=":dna:",
    layout="wide"
)

THEME_CSS = """
<style>
:root {
    color-scheme: light;
    --page-bg:
        linear-gradient(135deg,
            rgba(255, 241, 246, 0.98) 0%,
            rgba(238, 252, 244, 0.97) 32%,
            rgba(237, 247, 255, 0.97) 66%,
            rgba(248, 241, 255, 0.98) 100%);
    --hero-bg:
        linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.66)),
        linear-gradient(90deg, rgba(174, 226, 219, 0.30), rgba(255, 206, 219, 0.26), rgba(211, 203, 246, 0.24));
    --sidebar-bg:
        linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(246, 251, 255, 0.72));
    --glass-bg: rgba(255, 255, 255, 0.74);
    --glass-strong: rgba(255, 255, 255, 0.92);
    --glass-border: rgba(177, 199, 219, 0.34);
    --ink: #334155;
    --heading: #25324a;
    --muted: #6b7c93;
    --surface-subtle: rgba(255, 255, 255, 0.70);
    --surface-code: rgba(255, 255, 255, 0.82);
    --input-bg: rgba(255, 255, 255, 0.84);
    --button-bg: linear-gradient(135deg, rgba(174, 226, 219, 0.98), rgba(255, 206, 219, 0.98), rgba(255, 232, 176, 0.95));
    --button-text: #263449;
    --button-disabled-bg: rgba(255, 255, 255, 0.58);
    --button-disabled-text: rgba(51, 65, 85, 0.42);
    --button-border: rgba(166, 189, 212, 0.42);
    --button-border-hover: rgba(125, 156, 187, 0.56);
    --badge-text: #40706c;
    --canvas-border: rgba(177, 199, 219, 0.36);
    --teal: #aee2db;
    --coral: #ffcedb;
    --mint: #c9efd9;
    --lilac: #d3cbf6;
    --butter: #ffe8b0;
    --shadow: 0 18px 44px rgba(108, 121, 150, 0.13);
}

@media (prefers-color-scheme: dark) {
    :root {
        color-scheme: dark;
        --page-bg:
            linear-gradient(135deg,
                rgba(7, 8, 12, 0.99) 0%,
                rgba(16, 18, 26, 0.99) 34%,
                rgba(13, 23, 25, 0.98) 68%,
                rgba(24, 20, 32, 0.99) 100%);
        --hero-bg:
            linear-gradient(135deg, rgba(25, 28, 38, 0.94), rgba(11, 13, 18, 0.82)),
            linear-gradient(90deg, rgba(174, 226, 219, 0.16), rgba(255, 206, 219, 0.13), rgba(211, 203, 246, 0.14));
        --sidebar-bg:
            linear-gradient(180deg, rgba(22, 25, 35, 0.92), rgba(14, 20, 24, 0.82));
        --glass-bg: rgba(20, 23, 31, 0.82);
        --glass-strong: rgba(27, 31, 41, 0.94);
        --glass-border: rgba(214, 226, 240, 0.22);
        --ink: #f3f6fb;
        --heading: #ffffff;
        --muted: #c6d1de;
        --surface-subtle: rgba(30, 35, 45, 0.78);
        --surface-code: rgba(11, 14, 20, 0.86);
        --input-bg: rgba(16, 20, 29, 0.88);
        --button-bg: linear-gradient(135deg, rgba(174, 226, 219, 0.94), rgba(255, 206, 219, 0.94), rgba(255, 232, 176, 0.90));
        --button-text: #101720;
        --button-disabled-bg: rgba(48, 54, 67, 0.66);
        --button-disabled-text: rgba(243, 246, 251, 0.48);
        --button-border: rgba(214, 226, 240, 0.28);
        --button-border-hover: rgba(244, 248, 255, 0.48);
        --badge-text: #d8fff8;
        --canvas-border: rgba(214, 226, 240, 0.28);
        --teal: #aee2db;
        --coral: #ffcedb;
        --mint: #c9efd9;
        --lilac: #d3cbf6;
        --butter: #ffe8b0;
        --shadow: 0 18px 48px rgba(0, 0, 0, 0.44);
    }
}

@media (prefers-contrast: more) {
    :root {
        --glass-border: rgba(51, 76, 96, 0.58);
        --muted: #3b4d5d;
        --button-border: rgba(31, 52, 68, 0.62);
        --button-border-hover: rgba(20, 39, 54, 0.82);
        --canvas-border: rgba(31, 52, 68, 0.62);
        --shadow: 0 18px 46px rgba(43, 57, 78, 0.24);
    }
}

@media (prefers-color-scheme: dark) and (prefers-contrast: more) {
    :root {
        --glass-border: rgba(245, 250, 255, 0.58);
        --muted: #eef6ff;
        --button-border: rgba(245, 250, 255, 0.62);
        --button-border-hover: rgba(255, 255, 255, 0.82);
        --canvas-border: rgba(245, 250, 255, 0.62);
        --shadow: 0 18px 50px rgba(0, 0, 0, 0.60);
    }
}

.stApp {
    color: var(--ink);
    background: var(--page-bg);
    background-attachment: fixed;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.main .block-container {
    max-width: 1360px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

.app-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.2rem;
    padding: 1.25rem 1.4rem;
    border: 1px solid var(--glass-border);
    border-left: 6px solid var(--teal);
    border-radius: 8px;
    background: var(--hero-bg);
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
}

.app-hero h1 {
    margin: 0;
    color: var(--heading);
    font-size: 2.65rem;
    line-height: 1.08;
    letter-spacing: 0;
}

.app-hero p {
    margin: 0.45rem 0 0;
    color: var(--muted);
    font-size: 1.02rem;
    letter-spacing: 0;
}

.app-hero__badge {
    flex: 0 0 auto;
    min-width: 9rem;
    padding: 0.7rem 0.9rem;
    border: 1px solid rgba(174, 226, 219, 0.72);
    border-radius: 8px;
    background: var(--surface-subtle);
    color: var(--badge-text);
    font-weight: 700;
    text-align: center;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
h1,
h2,
h3 {
    color: var(--heading);
    letter-spacing: 0;
}

[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
label,
p,
li {
    color: var(--ink);
}

h2, h3 {
    margin-top: 0.65rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    background: var(--glass-bg);
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
}

div[data-testid="stFileUploader"] section {
    border: 1px dashed var(--glass-border);
    border-radius: 8px;
    background: var(--surface-subtle);
}

.canvas-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.15rem 0 0.85rem;
}

.canvas-meta span {
    display: inline-flex;
    align-items: center;
    min-height: 2.15rem;
    padding: 0.45rem 0.65rem;
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    background: var(--surface-subtle);
    color: var(--muted);
    font-size: 0.92rem;
}

.canvas-meta strong {
    color: var(--ink);
    margin-left: 0.3rem;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.55rem;
    border: 1px solid var(--button-border);
    border-radius: 8px;
    background: var(--button-bg);
    color: var(--button-text);
    font-weight: 750;
    box-shadow: var(--shadow);
    transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--button-border-hover);
    box-shadow: var(--shadow);
    transform: translateY(-1px);
}

.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: var(--button-disabled-bg);
    color: var(--button-disabled-text);
    box-shadow: none;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stSlider div[data-testid="stTickBar"],
.stColorPicker label + div {
    border-radius: 8px;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border: 1px solid var(--glass-border);
    background: var(--input-bg);
    color: var(--ink);
}

[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
pre,
code {
    border-radius: 8px;
}

[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow);
    overflow: hidden;
}

pre {
    border: 1px solid var(--glass-border);
    background: var(--surface-code) !important;
    color: var(--ink) !important;
}

canvas {
    border-radius: 8px;
    border: 1px solid var(--canvas-border);
    box-shadow: var(--shadow);
}

[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid var(--glass-border);
    background: var(--surface-code);
    color: var(--ink);
}

details {
    border-radius: 8px !important;
}

@media (max-width: 700px) {
    .main .block-container {
        padding-top: 1rem;
    }

    .app-hero {
        display: block;
        padding: 1rem;
    }

    .app-hero h1 {
        font-size: 2.05rem;
    }

    .app-hero__badge {
        margin-top: 0.85rem;
        text-align: left;
    }
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <section class="app-hero">
        <div>
            <h1>Meinlab Image Annotator</h1>
            <p>YOLO image annotation workspace</p>
        </div>
        <div class="app-hero__badge">Ready</div>
    </section>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================
if "classes" not in st.session_state:
    st.session_state.classes = [
        {"class_id": 0, "class_name": "artifact", "color": "#00FF00"},
        {"class_id": 1, "class_name": "egg", "color": "#FF0000"},
    ]

if "object_meta" not in st.session_state:
    st.session_state.object_meta = {}


# ============================================================
# Utility functions
# ============================================================
def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    normalized = hex_color.strip().lstrip("#")

    if len(normalized) == 3:
        normalized = "".join([channel * 2 for channel in normalized])

    try:
        red = int(normalized[0:2], 16)
        green = int(normalized[2:4], 16)
        blue = int(normalized[4:6], 16)
    except (TypeError, ValueError):
        red, green, blue = 123, 199, 196

    alpha = max(0, min(alpha, 1))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def resize_for_display(
    image: Image.Image,
    max_width: int
) -> Tuple[Image.Image, float]:
    """
    Resize image for display on canvas.

    Returns:
        display_image
        display_scale = display_width / original_width
    """
    original_w, original_h = image.size

    if original_w <= max_width:
        return image.copy(), 1.0

    scale = max_width / original_w
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    display_image = image.resize((new_w, new_h))
    return display_image, scale


def make_object_key(obj: Dict[str, Any]) -> str:
    """
    Create a stable-ish key from canvas object geometry.

    This is used to store class_id and annotation_type per drawn object.
    If the object is moved or resized, the key may change, so the app will
    treat it as a new object. This is acceptable for a simple annotator.
    """
    essential = {
        "type": obj.get("type"),
        "left": round(float(obj.get("left", 0)), 2),
        "top": round(float(obj.get("top", 0)), 2),
        "width": round(float(obj.get("width", 0)), 2),
        "height": round(float(obj.get("height", 0)), 2),
        "scaleX": round(float(obj.get("scaleX", 1)), 4),
        "scaleY": round(float(obj.get("scaleY", 1)), 4),
        "points": obj.get("points", None),
    }

    key_string = json.dumps(essential, sort_keys=True)
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()


def rect_to_original_coordinates(
    obj: Dict[str, Any],
    display_scale: float
) -> Tuple[float, float, float, float]:
    """
    Convert Fabric.js rectangle coordinates from display image space
    back to original image coordinates.
    """
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    width = float(obj.get("width", 0))
    height = float(obj.get("height", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))

    x1_display = left
    y1_display = top
    x2_display = left + width * scale_x
    y2_display = top + height * scale_y

    x1 = x1_display / display_scale
    y1 = y1_display / display_scale
    x2 = x2_display / display_scale
    y2 = y2_display / display_scale

    return x1, y1, x2, y2


def polygon_to_original_coordinates(
    obj: Dict[str, Any],
    display_scale: float
) -> List[Tuple[float, float]]:
    """
    Convert Fabric.js polygon coordinates from display image space
    back to original image coordinates.
    """
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))

    raw_points = obj.get("points", [])
    points = []

    for point in raw_points:
        px = float(point.get("x", 0))
        py = float(point.get("y", 0))

        x_display = left + px * scale_x
        y_display = top + py * scale_y

        x_original = x_display / display_scale
        y_original = y_display / display_scale

        points.append((x_original, y_original))

    return points


def clip_point(
    x: float,
    y: float,
    image_w: int,
    image_h: int
) -> Tuple[float, float]:
    x = max(0, min(x, image_w))
    y = max(0, min(y, image_h))
    return x, y


def normalize_bbox_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int
) -> Tuple[float, float, float, float]:
    """
    YOLO detection format:

    class_id x_center y_center width height

    All coordinates are normalized between 0 and 1.
    """
    x1, y1 = clip_point(x1, y1, image_w, image_h)
    x2, y2 = clip_point(x2, y2, image_w, image_h)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    box_w = xmax - xmin
    box_h = ymax - ymin

    x_center = xmin + box_w / 2
    y_center = ymin + box_h / 2

    return (
        x_center / image_w,
        y_center / image_h,
        box_w / image_w,
        box_h / image_h,
    )


def normalize_polygon_yolo(
    points: List[Tuple[float, float]],
    image_w: int,
    image_h: int
) -> List[float]:
    """
    YOLO segmentation format:

    class_id x1 y1 x2 y2 x3 y3 ...

    All coordinates are normalized between 0 and 1.
    """
    normalized = []

    for x, y in points:
        x, y = clip_point(x, y, image_w, image_h)
        normalized.append(x / image_w)
        normalized.append(y / image_h)

    return normalized


def build_yolo_label_text(annotation_rows: List[Dict[str, Any]]) -> str:
    lines = []

    for row in annotation_rows:
        class_id = row["class_id"]

        if row["annotation_type"] in ["rectangle", "crop image"]:
            x_center, y_center, w, h = row["yolo_values"]
            line = (
                f"{class_id} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{w:.6f} {h:.6f}"
            )
            lines.append(line)

        elif row["annotation_type"] == "polygon":
            polygon_values = " ".join(
                [f"{v:.6f}" for v in row["yolo_values"]]
            )
            line = f"{class_id} {polygon_values}"
            lines.append(line)

    return "\n".join(lines)


def draw_annotations_on_image(
    image: Image.Image,
    annotation_rows: List[Dict[str, Any]],
    class_config: Dict[int, Dict[str, str]]
) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)

    for row in annotation_rows:
        class_id = row["class_id"]
        class_name = row["class_name"]
        color = class_config[class_id]["color"]

        if row["annotation_type"] in ["rectangle", "crop image"]:
            x1, y1, x2, y2 = row["absolute_points"]
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline=color,
                width=4
            )
            draw.text(
                (xmin, max(0, ymin - 18)),
                f"{class_id}: {class_name}",
                fill=color
            )

        elif row["annotation_type"] == "polygon":
            points = row["absolute_points"]

            if len(points) >= 3:
                draw.line(
                    points + [points[0]],
                    fill=color,
                    width=4
                )
                draw.text(
                    points[0],
                    f"{class_id}: {class_name}",
                    fill=color
                )

    return output


def crop_regions(
    image: Image.Image,
    annotation_rows: List[Dict[str, Any]]
) -> Dict[str, bytes]:
    crop_outputs = {}

    for i, row in enumerate(annotation_rows):
        if row["annotation_type"] != "crop image":
            continue

        x1, y1, x2, y2 = row["absolute_points"]

        xmin, xmax = sorted([int(x1), int(x2)])
        ymin, ymax = sorted([int(y1), int(y2)])

        if xmax <= xmin or ymax <= ymin:
            continue

        crop = image.crop((xmin, ymin, xmax, ymax))
        crop_name = f"crop_{i:03d}_class{row['class_id']}_{row['class_name']}.png"
        crop_outputs[crop_name] = pil_to_png_bytes(crop)

    return crop_outputs


def build_zip_package(
    base_filename: str,
    annotated_image: Image.Image,
    label_text: str,
    annotation_table: pd.DataFrame,
    crop_outputs: Dict[str, bytes]
) -> bytes:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{base_filename}_annotated.png",
            pil_to_png_bytes(annotated_image)
        )

        zf.writestr(
            f"{base_filename}.txt",
            text_to_bytes(label_text)
        )

        zf.writestr(
            f"{base_filename}_annotations.csv",
            annotation_table.to_csv(index=False)
        )

        for crop_name, crop_bytes in crop_outputs.items():
            zf.writestr(f"crops/{crop_name}", crop_bytes)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================
# Class configuration
# ============================================================
with st.container(border=True):
    st.header("1. Class Configuration")

    num_classes = st.number_input(
        "Number of classes",
        min_value=1,
        max_value=50,
        value=len(st.session_state.classes),
        step=1
    )

    while len(st.session_state.classes) < num_classes:
        new_id = len(st.session_state.classes)
        st.session_state.classes.append(
            {
                "class_id": new_id,
                "class_name": f"class_{new_id}",
                "color": "#FFFF00",
            }
        )

    while len(st.session_state.classes) > num_classes:
        st.session_state.classes.pop()

    for i in range(num_classes):
        with st.expander(f"Class {i}", expanded=True):
            id_col, name_col, color_col = st.columns([1, 2, 1])

            with id_col:
                st.session_state.classes[i]["class_id"] = st.number_input(
                    f"Class label ID {i}",
                    min_value=0,
                    max_value=999,
                    value=int(st.session_state.classes[i]["class_id"]),
                    step=1,
                    key=f"class_id_{i}"
                )

            with name_col:
                st.session_state.classes[i]["class_name"] = st.text_input(
                    f"Class name {i}",
                    value=st.session_state.classes[i]["class_name"],
                    key=f"class_name_{i}"
                )

            with color_col:
                st.session_state.classes[i]["color"] = st.color_picker(
                    f"Draw color {i}",
                    value=st.session_state.classes[i]["color"],
                    key=f"class_color_{i}"
                )

class_config = {}

for item in st.session_state.classes:
    class_id = int(item["class_id"])

    class_config[class_id] = {
        "name": item["class_name"],
        "color": item["color"],
    }

class_options = {
    f"{class_id} - {info['name']}": class_id
    for class_id, info in class_config.items()
}


# ============================================================
# Image upload
# ============================================================
with st.container(border=True):
    st.header("2. Upload Image")
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )


# ============================================================
# Annotation configuration
# ============================================================
with st.container(border=True):
    st.header("3. Annotation Settings")
    settings_col, status_col = st.columns([2, 1], gap="large")

    with settings_col:
        selected_class_label = st.selectbox(
            "Active class for newly drawn objects",
            options=list(class_options.keys())
        )

        selected_annotation_type = st.selectbox(
            "Annotation type for newly drawn objects",
            options=["rectangle", "polygon", "crop image"]
        )

        control_col1, control_col2 = st.columns(2)

        with control_col1:
            stroke_width = st.slider(
                "Stroke width",
                min_value=1,
                max_value=10,
                value=3
            )

        with control_col2:
            display_width = st.slider(
                "Display width",
                min_value=400,
                max_value=1600,
                value=1000,
                step=100
            )

        if st.button("Reset object metadata"):
            st.session_state.object_meta = {}
            st.success("Object metadata has been reset.")

        st.info(
            "Class ID and annotation type are saved per drawn object. "
            "Changing the active class will only affect new objects."
        )

    selected_class_id = class_options[selected_class_label]
    selected_class_name = class_config[selected_class_id]["name"]
    selected_color = class_config[selected_class_id]["color"]

    if selected_annotation_type == "polygon":
        drawing_mode = "polygon"
    else:
        drawing_mode = "rect"

    with status_col:
        st.subheader("Current Drawing Mode")
        st.markdown(f"**New object class ID:** `{selected_class_id}`")
        st.markdown(f"**New object class name:** `{selected_class_name}`")
        st.markdown(f"**New object type:** `{selected_annotation_type}`")
        st.color_picker(
            "Active draw color",
            value=selected_color,
            disabled=True
        )

        st.subheader("YOLO Format")
        if selected_annotation_type in ["rectangle", "crop image"]:
            st.code("class_id x_center y_center width height", language="text")
        else:
            st.code("class_id x1 y1 x2 y2 x3 y3 ...", language="text")

if uploaded_file is None:
    st.warning("Please upload an image to start annotation.")
    st.stop()

original_image = Image.open(uploaded_file).convert("RGB")
original_w, original_h = original_image.size

display_image, display_scale = resize_for_display(
    original_image,
    max_width=display_width
)

display_w, display_h = display_image.size

base_filename = uploaded_file.name.rsplit(".", 1)[0]


# ============================================================
# Canvas
# ============================================================
canvas_fill_color = hex_to_rgba(selected_color, 0.18)

with st.container(border=True):
    st.subheader("3.1 Image Annotation Canvas")
    st.markdown(
        f"""
        <div class="canvas-meta">
            <span>Original <strong>{original_w} x {original_h}</strong></span>
            <span>Display <strong>{display_w} x {display_h}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    canvas_result = st_canvas(
        fill_color=canvas_fill_color,
        stroke_width=stroke_width,
        stroke_color=selected_color,
        background_image=display_image,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode=drawing_mode,
        key="annotation_canvas",
    )


# ============================================================
# Parse canvas objects and preserve class_id per object
# ============================================================
raw_objects = []

if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])

    for idx, obj in enumerate(objects):
        obj_type = obj.get("type", "")

        if obj_type not in ["rect", "polygon"]:
            continue

        object_key = make_object_key(obj)

        if object_key not in st.session_state.object_meta:
            st.session_state.object_meta[object_key] = {
                "class_id": selected_class_id,
                "annotation_type": selected_annotation_type,
            }

        object_meta = st.session_state.object_meta[object_key]

        raw_objects.append(
            {
                "index": idx,
                "object_key": object_key,
                "canvas_type": obj_type,
                "class_id": int(object_meta["class_id"]),
                "annotation_type": object_meta["annotation_type"],
                "object": obj,
            }
        )


# ============================================================
# Editable metadata table
# ============================================================
st.subheader("3.2 Annotation Class and Type Assignment")

if len(raw_objects) > 0:
    meta_records = []

    for item in raw_objects:
        meta_records.append(
            {
                "index": item["index"],
                "object_key": item["object_key"],
                "canvas_type": item["canvas_type"],
                "class_id": item["class_id"],
                "annotation_type": item["annotation_type"],
            }
        )

    meta_df = pd.DataFrame(meta_records)

    edited_meta_df = st.data_editor(
        meta_df,
        use_container_width=True,
        column_config={
            "class_id": st.column_config.SelectboxColumn(
                "class_id",
                options=sorted(list(class_config.keys())),
                required=True
            ),
            "annotation_type": st.column_config.SelectboxColumn(
                "annotation_type",
                options=["rectangle", "polygon", "crop image"],
                required=True
            ),
            "object_key": st.column_config.TextColumn(
                "object_key",
                disabled=True
            ),
        },
        disabled=["index", "object_key", "canvas_type"],
        key="annotation_metadata_editor"
    )

    for _, row in edited_meta_df.iterrows():
        object_key = row["object_key"]
        class_id = int(row["class_id"])
        annotation_type = row["annotation_type"]

        st.session_state.object_meta[object_key] = {
            "class_id": class_id,
            "annotation_type": annotation_type,
        }

else:
    edited_meta_df = pd.DataFrame()
    st.info("No annotation objects yet. Draw on the image canvas.")


# ============================================================
# Convert canvas objects to final annotation rows
# ============================================================
annotation_rows = []

for item in raw_objects:
    object_key = item["object_key"]
    obj = item["object"]

    meta = st.session_state.object_meta.get(
        object_key,
        {
            "class_id": selected_class_id,
            "annotation_type": selected_annotation_type,
        }
    )

    class_id = int(meta["class_id"])

    if class_id not in class_config:
        continue

    class_name = class_config[class_id]["name"]
    annotation_type = meta["annotation_type"]
    canvas_type = item["canvas_type"]

    if annotation_type in ["rectangle", "crop image"]:
        if canvas_type != "rect":
            continue

        x1, y1, x2, y2 = rect_to_original_coordinates(
            obj,
            display_scale
        )

        yolo_values = normalize_bbox_yolo(
            x1,
            y1,
            x2,
            y2,
            image_w=original_w,
            image_h=original_h
        )

        annotation_rows.append(
            {
                "index": item["index"],
                "object_key": object_key,
                "annotation_type": annotation_type,
                "class_id": class_id,
                "class_name": class_name,
                "absolute_points": (x1, y1, x2, y2),
                "yolo_values": yolo_values,
            }
        )

    elif annotation_type == "polygon":
        if canvas_type != "polygon":
            continue

        points = polygon_to_original_coordinates(
            obj,
            display_scale
        )

        if len(points) < 3:
            continue

        yolo_values = normalize_polygon_yolo(
            points,
            image_w=original_w,
            image_h=original_h
        )

        annotation_rows.append(
            {
                "index": item["index"],
                "object_key": object_key,
                "annotation_type": annotation_type,
                "class_id": class_id,
                "class_name": class_name,
                "absolute_points": points,
                "yolo_values": yolo_values,
            }
        )


# ============================================================
# Annotation output table
# ============================================================
st.subheader("3.3Annotation Output Table")

table_records = []

for row in annotation_rows:
    if row["annotation_type"] in ["rectangle", "crop image"]:
        x1, y1, x2, y2 = row["absolute_points"]
        x_center, y_center, w, h = row["yolo_values"]

        table_records.append(
            {
                "index": row["index"],
                "type": row["annotation_type"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "yolo": (
                    f"{row['class_id']} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{w:.6f} {h:.6f}"
                ),
            }
        )

    elif row["annotation_type"] == "polygon":
        yolo_values = " ".join(
            [f"{v:.6f}" for v in row["yolo_values"]]
        )

        table_records.append(
            {
                "index": row["index"],
                "type": row["annotation_type"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "num_points": len(row["absolute_points"]),
                "yolo": f"{row['class_id']} {yolo_values}",
            }
        )

annotation_table = pd.DataFrame(table_records)

if len(annotation_table) > 0:
    st.dataframe(annotation_table, use_container_width=True)
else:
    st.info("No valid annotations generated yet.")


# ============================================================
# Generate outputs
# ============================================================
label_text = build_yolo_label_text(annotation_rows)

annotated_image = draw_annotations_on_image(
    original_image,
    annotation_rows,
    class_config=class_config
)

crop_outputs = crop_regions(
    original_image,
    annotation_rows
)


# ============================================================
# Preview YOLO label
# ============================================================
st.subheader("3.4 YOLO Label Preview")

if label_text.strip():
    st.code(label_text, language="text")
else:
    st.code("# No YOLO annotation generated yet.", language="text")


# ============================================================
# Preview annotated image
# ============================================================
st.subheader("3.5 Annotated Image Preview")
st.image(
    annotated_image,
    caption="Annotated image",
    use_container_width=True
)


# ============================================================
# Download outputs
# ============================================================
st.subheader("3.6 Download Outputs")

zip_bytes = build_zip_package(
    base_filename=base_filename,
    annotated_image=annotated_image,
    label_text=label_text,
    annotation_table=annotation_table,
    crop_outputs=crop_outputs
)

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="Download YOLO TXT",
        data=text_to_bytes(label_text),
        file_name=f"{base_filename}.txt",
        mime="text/plain",
        disabled=len(annotation_rows) == 0
    )

with download_col2:
    st.download_button(
        label="Download Annotated Image",
        data=pil_to_png_bytes(annotated_image),
        file_name=f"{base_filename}_annotated.png",
        mime="image/png",
        disabled=len(annotation_rows) == 0
    )

with download_col3:
    st.download_button(
        label="Download ZIP Package",
        data=zip_bytes,
        file_name=f"{base_filename}_yolo_annotation_package.zip",
        mime="application/zip",
        disabled=len(annotation_rows) == 0
    )


# ============================================================
# Crop preview
# ============================================================
if len(crop_outputs) > 0:
    st.subheader("Crop Image Outputs")

    for crop_name, crop_bytes in crop_outputs.items():
        crop_image = Image.open(io.BytesIO(crop_bytes))
        st.image(
            crop_image,
            caption=crop_name,
            use_container_width=False
        )


# ============================================================
# Debug information
# ============================================================
with st.expander("Debug: Canvas JSON"):
    st.json(canvas_result.json_data)

with st.expander("Debug: Object Metadata"):
    st.json(st.session_state.object_meta)
