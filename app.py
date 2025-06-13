import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="Parking Lot Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🅿️ Parking Lot Status Detection")
st.write(
    "Upload the parking lot image to detect wether or not if it's empty or filled "
)

# Dropdown for selecting YOLO model version
model_options = {
    "YOLOv8n": "best8.pt", 
    "YOLOv11n": "best11.pt", 
    "YOLOv12n": "best12.pt"
}
selected_model_name = st.selectbox("Choose Detection Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]  # Get the corresponding .pt file path

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        # model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Model Failed to Load: {e}")
        return None

def detect_and_draw_on_image(
    input_image: Image.Image, 
    model: YOLO,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> (np.ndarray, dict):
    
    results = model.predict(
        source=input_image,
        conf=conf_threshold,
        iou=iou_threshold
    )
    

    annotated_image_bgr = np.array(input_image.convert('RGB'))
    annotated_image_bgr = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_RGB2BGR)

    stats = {'empty': 0, 'filled': 0}
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])

        if class_id == 1: # 1: filled
            color = (0, 0, 255) 
            stats['filled'] += 1
        else: # 0: empty
            color = (0, 255, 0) 
            stats['empty'] += 1
            
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        
        cv2.rectangle(
            img=annotated_image_bgr, 
            pt1=(x1, y1), 
            pt2=(x2, y2), 
            color=color, 
            thickness=1 
        )


    stats['total'] = stats['empty'] + stats['filled']

    #Used to draw stats onto the detected image
    # def draw_text_with_background(img, text, position, font, scale, text_color, bg_color, thickness=2, padding=10, alpha=0.6):
    #     overlay = img.copy()
    #     text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    #     text_w, text_h = text_size
    #     x, y = position
    #     cv2.rectangle(overlay, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), bg_color, -1)
    #     cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    #     cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # draw_text_with_background(annotated_image_bgr, f"Filled Spots: {stats['filled']}", (30, 50), font, 1, (0, 0, 255), (255,255,255))
    # draw_text_with_background(annotated_image_bgr, f"Empty Spots: {stats['empty']}", (30, 100), font, 1, (0, 255, 0), (255,255,255))
    # draw_text_with_background(annotated_image_bgr, f"Total Spots: {stats['total']}", (30, 150), font, 1, (0,0,0), (255,255,255))

    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

    return annotated_image_rgb, stats

model = load_model(selected_model_path)

if 'final_image' not in st.session_state:
    st.session_state.final_image = None

if 'stats' not in st.session_state:
    st.session_state.stats = None

#Upload File
uploaded_file = st.file_uploader(
    "Upload an image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3, col4, col5 = st.columns([1, 3.5, 1, 3.5, 1])
    col123 = st.columns(2)
    col11,col12 = st.columns([1, 1])

    with col2:
        header_col, button_col = st.columns([3, 1])

        with header_col:
            st.subheader("Original Image")

        with button_col:
            st.write("") 
            start_detection = st.button("Detect!", use_container_width=True)

        st.image(image, caption="Image that you've uploaded.", use_column_width=True)

    with col2:
            st.subheader("Detection Settings")
            conf_threshold = st.slider(
                "TConfidence Threshold",
                min_value=0.0, max_value=1.0, value=0.25, step=0.01
            )
            iou_threshold = st.slider(
                "IoU Threshold",
                min_value=0.0, max_value=1.0, value=0.45, step=0.01
            )

    if start_detection:
        if model is not None:
            # Loading circle
            with col3:
                with st.spinner('Memproses...'):
                    final_image, stats = detect_and_draw_on_image(
                        input_image=image,
                        model=model,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold
                    )
                    st.session_state.final_image = final_image
                    st.session_state.stats = stats
        else:
            st.error("Model failed to load.")

    if st.session_state.final_image is not None:
        with col4:
            st.subheader("Detection Result")
            st.image(st.session_state.final_image, caption="Image With Detection.", use_column_width=True)
            
            st.divider()
            stats = st.session_state.stats
            m_col1, m_col2, m_col3 = st.columns(3)
            # with m_col1:
            #     st.metric(label="Total Slots", value=stats['total'])
            # with m_col2:
            #     st.metric(label="Filled", value=stats['filled'])
            # with m_col3:
            #     st.metric(label="Empty", value=stats['empty'])
            with m_col1:
                st.markdown(f"<p style='color: black; font-size: 20px;'>Total Slots: {stats['total']}</p>", unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"<p style='color: red; font-size: 20px;'>Filled: {stats['filled']}</p>", unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"<p style='color: green; font-size: 20px;'>Empty: {stats['empty']}</p>", unsafe_allow_html=True)
    else:
        st.info("Please upload an image to start.")

else:
    st.info("Pleae upload an image to start.")
