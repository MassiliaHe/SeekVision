import os
import cv2
import tempfile
import numpy as np
import streamlit as st
import supervision as sv
from dotenv import load_dotenv
from pathlib import Path
from pycocotools import mask as mask_utils

# DDS Cloud API v2 imports
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task

# Load environment variables from the .env file (if present)
load_dotenv()

def run_inference(api_token: str, text_prompt: str, bbox_threshold: float, image_bytes: bytes, prompt_free: bool = False):
    """
    Runs the DDS Cloud API inference (V2) on the provided image bytes and returns the annotated image.
    """
    # Write the uploaded image bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    # Convert the image to base64 as required by the new API
    image_base64 = image_to_base64(temp_file_path)
    os.remove(temp_file_path)

    # Initialize DDS Cloud API client
    config = Config(api_token)
    client = Client(config)

    # Build API body
    api_body = {
        "model": "DINO-X-1.0",
        "image": image_base64,
        "prompt": {
            "type": "universal"
        },
        "targets": ["bbox", "mask"],
        "mask_format": "coco_rle",
        "bbox_threshold": bbox_threshold,
        "iou_threshold": 0.8
    }
    # Add text to the prompt if not in prompt-free mode
    if not prompt_free and text_prompt.strip():
        api_body["prompt"]["text"] = text_prompt.strip()

    # Create and run the V2Task
    task = V2Task(
        api_path="/v2/task/dinox/detection",
        api_body=api_body
    )
    client.run_task(task)
    result = task.result

    objects = result.get("objects", [])

    # Prepare class mappings and JSON result
    classes = [obj["category"].lower().strip() for obj in objects]
    class_name_to_id = {name: idx for idx, name in enumerate(set(classes))}

    boxes = []
    masks = []
    confidences = []
    class_names = []
    class_ids = []
    json_result = {"detections": []}

    for obj in objects:
        boxes.append(obj["bbox"])
        masks.append(mask_utils.decode(obj["mask"]))
        confidences.append(obj["score"])
        cls_name = obj["category"].lower().strip()
        class_names.append(cls_name)
        class_ids.append(class_name_to_id[cls_name])
        json_result["detections"].append({
            "bbox": obj["bbox"],  # [x_min, y_min, x_max, y_max]
            "category_name": cls_name,
            "category_id": class_name_to_id[cls_name],
            "score": obj["score"]
        })

    boxes = np.array(boxes)
    masks = np.array(masks)
    class_ids = np.array(class_ids)
    labels = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]

    # Decode image for annotation
    file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if not len(boxes):
        return img.copy(), json_result

    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool),
        class_id=class_ids,
    )

    # Annotate the image
    annotated_img = img.copy()
    box_annotator = sv.BoxAnnotator()
    annotated_img = box_annotator.annotate(scene=annotated_img, detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_img = label_annotator.annotate(
        scene=annotated_img, detections=detections, labels=labels
    )
    mask_annotator = sv.MaskAnnotator()
    annotated_img = mask_annotator.annotate(scene=annotated_img, detections=detections)

    return annotated_img, json_result

def main():
    st.set_page_config(page_title="SeekVision", page_icon="icons/SeekVision_small.png")
    st.title("DINO-X Inference with DDS Cloud API (v2)")

    # Sidebar: Logo + Nom d'app + Configuration
    with st.sidebar:
        # Affiche le logo et le nom côte à côte
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("icons/SeekVision.png", width=40)
        with col2:
            st.markdown(
                "<h2 style='margin-bottom: 0; margin-top: 10px; font-weight: 800; letter-spacing: 1px;'>SeekVision</h2>",
                unsafe_allow_html=True
            )
        st.markdown("---")  # ligne de séparation élégante
        st.header("Configuration")
        API_KEY = os.getenv('API_KEY') or ""
        api_token = st.text_input(
            "API Token", type="password", value=API_KEY
        )
        text_prompt = st.text_input("Text Prompt", value="Text . logo . image")
        bbox_threshold = st.slider(
            "BBox Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.05
        )
        prompt_free = st.checkbox("Prompt-Free Mode", value=False, help="Ignore the text prompt and use universal detection.")

    # Main: Upload an image
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if st.button("Run Inference"):
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()

            # Decode the uploaded image for display
            file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run the inference using your custom function
            with st.spinner("Running inference..."):
                annotated_img, result_json = run_inference(
                    api_token, text_prompt, bbox_threshold, image_bytes, prompt_free
                )
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            # Container for side-by-side image display
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(img_rgb, use_container_width=True)
                with col2:
                    st.subheader("Annotated Image")
                    st.image(annotated_rgb, use_container_width=True)

            # Container for nicely displaying the JSON output
            with st.container(border=True):
                st.subheader("Inference Output (JSON)")
                st.json(result_json)
        else:
            st.warning("Please add an image to run DinoX Inference")

if __name__ == "__main__":
    main()
