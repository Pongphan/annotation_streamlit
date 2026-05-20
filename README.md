# Simple Streamlit YOLO Image Annotator

A simple Streamlit web app for image annotation and YOLO-format export.

## Features

- Upload image
- Display image
- Configure class ID, class name, and annotation color
- Select annotation type:
  - rectangle
  - polygon
  - crop image
- Draw annotation overlay
- Show annotation output table
- Download:
  - YOLO TXT label
  - annotated PNG image
  - ZIP package containing image, label, CSV, and crops

## YOLO Formats

### Rectangle / Detection

```txt
class_id x_center y_center width height