# Streamlit YOLO Image Annotator

A lightweight Streamlit app for image annotation and YOLO-format export.

## Features

- Upload JPG, PNG, BMP, TIFF, or TIF images.
- Draw rectangle, polygon, or crop annotations.
- Configure class IDs, names, and colors.
- Edit class/type metadata for drawn objects.
- Preview YOLO labels and the annotated image.
- Download YOLO TXT, annotated PNG, or a ZIP package with labels, CSV metadata, and crops.

## YOLO formats

Rectangle and crop annotations export as detection rows:

```txt
class_id x_center y_center width height
```

Polygon annotations export as segmentation rows:

```txt
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized between 0 and 1.

## Run locally

From the repository root:

```bash
pip install -r annotation_streamlit/requirements.txt
streamlit run annotation_streamlit/app.py
```

## Deploy on Streamlit Community Cloud

- Repository: select this repository.
- Main file path: `annotation_streamlit/app.py`
- Dependency file: Streamlit will install `annotation_streamlit/requirements.txt`.
- Config file: the repository-root `.streamlit/config.toml` is included for Cloud deployment.
