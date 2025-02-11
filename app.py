import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import supervision as sv
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

# DDS Cloud API imports
from dds_cloudapi_sdk import Config, Client, TextPrompt
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget


def run_inference(api_token: str, text_prompt: str, bbox_threshold: float, image_bytes: bytes):
    """
    Runs the DDS Cloud API inference on the provided image bytes and returns the annotated image.
    """
    # Write the uploaded image bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    # Initialize DDS Cloud API client
    config = Config(api_token)
    client = Client(config)

    # Upload file to the DDS server to get its URL
    image_url = client.upload_file(temp_file_path)

    # Create and run the DINO-X task
    task = DinoxTask(
        image_url=image_url,
        prompts=[TextPrompt(text=text_prompt)],
        bbox_threshold=bbox_threshold,
        targets=[DetectionTarget.BBox, DetectionTarget.Mask]
    )
    client.run_task(task)
    predictions = task.result.objects

    # Remove the temporary file since it is no longer needed
    os.remove(temp_file_path)

    # Prepare class mappings from the text prompt (e.g., "Text . logo . image")
    classes = [x.strip().lower() for x in text_prompt.split('.') if x]
    class_name_to_id = {name: idx for idx, name in enumerate(classes)}

    # Process predictions to extract boxes, masks, scores, and class IDs
    boxes = []
    masks = []
    confidences = []
    class_names = []
    class_ids = []
    json_result = {
        "detections": []
    }

    for obj in predictions:
        boxes.append(obj.bbox)
        x_min, y_min, x_max, y_max = obj.bbox
        width = x_max - x_min
        height = y_max - y_min
        # Convert the mask from RLE to a numpy array
        mask = DetectionTask.rle2mask(
            DetectionTask.string2rle(obj.mask.counts), obj.mask.size
        )
        masks.append(mask)
        confidences.append(obj.score)
        cls_name = obj.category.lower().strip()
        class_names.append(cls_name)
        # Use the mapping from the prompt; if not found, assign a default value (-1)
        class_ids.append(class_name_to_id.get(cls_name, -1))
            # Format JSON result
        json_result["detections"].append({
            "bbox": [x_min, y_min, width, height],  # COCO-style format
            "category_name": cls_name,
            "category_id": class_ids[-1],
            "score": confidences[-1]
        })

    boxes = np.array(boxes)
    masks = np.array(masks)
    class_ids = np.array(class_ids)
    labels = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]


    # Read the original image (using OpenCV) from the uploaded bytes for annotation
    file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    # Create a detections object for visualization using supervision
    if not len(boxes):
        return img.copy(), json_result
    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool),
        class_id=class_ids,
    )

    # Annotate the image: draw bounding boxes, labels, and masks
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
    st.title("DINO-X Inference with DDS Cloud API")

    # Sidebar: Configuration options for API key, prompt, and bbox threshold
    st.sidebar.header("Configuration")
    API_KEY = os.getenv('API_KEY') or ""
    api_token = st.sidebar.text_input(
        "API Token", type="password", value=API_KEY
    )
    text_prompt = st.sidebar.text_input("Text Prompt", value="Text . logo . image")
    bbox_threshold = st.sidebar.slider(
        "BBox Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.05
    )

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
                print("Inference ...")
                annotated_img, result_json = run_inference(
                    api_token, text_prompt, bbox_threshold, image_bytes
                )
                # result_json = {}
                # annotated_img = img 
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
            with st.container():
                st.subheader("Inference Output (JSON)")
                st.json(result_json)

        else:
            st.warning("Please add an image to run DinoX Inference")


if __name__ == "__main__":
    main()
