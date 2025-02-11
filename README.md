
# **Lasqo: DINO-X Inference with DDS Cloud API** 🚀  

Lasqo is a **Streamlit app** for object detection and segmentation using the **DDS Cloud API** and **DINO-X**.  
Upload an image, set a text prompt, and get **annotated results with bounding boxes and masks**.

---

## 📌 **Example Output**  

### 🖼️ **Original vs. Annotated Image**  
![Original](file:///E:/Workspace/LasqoApp/Assets/iphone-apps-app-store.jpg) | ![Annotated](file:///E:/Workspace/LasqoApp/results/iphone-apps-app-store.png)


## ⚙️ **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/lasqo.git
cd lasqo
```

### **2️⃣ Install Dependencies**  
```bash
uv pip install -r pyproject.toml
```

### **3️⃣ Set Up Environment Variables**  
Create a `.env` file and add your API key:  
```env
API_KEY=your_dds_api_key_here
```

### **4️⃣ Run the Application**  
```bash
streamlit run app.py
```

---

## 🎯 **Usage**  

1. **Enter your API key** or load it from `.env`.  
2. **Set a text prompt** (e.g., `"Text . logo . image"`).  
3. **Adjust the bounding box threshold** if needed.  
4. **Upload an image** (JPG, PNG).  
5. **Click "Run Inference"** to get results.  

---

## 📊 **Output Format**  

### **📌 Annotated Image**  
Displays bounding boxes, masks, and labels.  

### **📌 JSON Output (COCO-Style)**  
```json
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
```

---

## 🛠 **Tech Stack**  
- **Streamlit** – UI  
- **DDS Cloud API** – Object Detection  
- **Supervision** – Visualization  
- **OpenCV, NumPy** – Image Processing  

---

## 📜 **License**  
MIT License.  

---

## ⭐ **Support the Project**  
If you find this useful, **give it a star ⭐ on GitHub!** 🎉
```

---

### **Steps to Complete Your README**
1. **Place Example Images in Your Repo**  
   - Add `original_image.jpg` and `annotated_image.jpg` inside an `assets/` folder.
   - Example command:
     ```bash
     mkdir assets
     cp example_original.jpg assets/original_image.jpg
     cp example_annotated.jpg assets/annotated_image.jpg
     ```

2. **Commit & Push the Changes**  
   ```bash
   git add README.md assets/original_image.jpg assets/annotated_image.jpg
   git commit -m "Added example images and README"
   git push origin main
   ```
