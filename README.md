# **SeekVision: DINO-X Detection & Segmentation with DDS Cloud API** 🎯

SeekVision is a **Streamlit application** for object detection and segmentation using the **DDS Cloud API** and **DINO-X** models.  
Upload your image, choose prompt or prompt-free mode, and get instant visual results: bounding boxes, masks, and class labels.

---

## 📌 **Example Output**

### 🖼️ Original vs. Annotated Image

| Original | Annotated |
| :------: | :-------: |
| ![Original](Assets/iphone-apps-app-store.jpg) | ![Annotated](assets/annotated_image.jpg) |

---

## ⚙️ **Installation & Setup**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/SeekVision.git
cd SeekVision
2️⃣ Install Dependencies
bash
Copier
Modifier
uv pip install -r pyproject.toml
Or use pip install -r requirements.txt if not using uv.

3️⃣ Set Up Environment Variables
Create a .env file and add your DDS Cloud API key:

env
Copier
Modifier
API_KEY=your_dds_api_key_here
4️⃣ Run the Application
bash
Copier
Modifier
uv run streamlit run app.py
🎯 Usage
Enter your API key or load it from .env.

Choose Prompt or Prompt-Free mode.

(Optional) Set a text prompt (e.g., Text . logo . image).

Adjust the bounding box threshold if needed.

Upload an image (.jpg, .png).

Click "Run Inference" and view results instantly.

📊 Output Format
🖼️ Annotated Image
Bounding boxes, masks, and class labels are displayed on your image.

📝 JSON Output (COCO-Style)
json
Copier
Modifier
{
    "detections": [
        {
            "bbox": [x_min, y_min, width, height],
            "category_name": "logo",
            "category_id": 1,
            "score": 0.85
        }
    ]
}
🛠 Tech Stack
Streamlit – User Interface

DDS Cloud API – Object Detection

Supervision – Visualization

OpenCV, NumPy – Image Processing

📜 License
MIT License

⭐ Support the Project
If you find this useful, give it a star ⭐ on GitHub! 🎉

