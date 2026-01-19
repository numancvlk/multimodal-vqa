# LIBRARIES
import requests
import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# HYPERPARAMETERS
URL = os.getenv("LOCAL_URL")

st.set_page_config(page_title="Multimodal AI Agent", layout="wide")

st.title("Multimodal VQA AI Agent")
st.markdown("---")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')
    else:
        st.info("Please upload an image to start.")

with col2:
    st.subheader("2. Ask Question")
    
    question = st.text_input("What do you want to know about the image?", placeholder="e.g., How many birds are there?")
    btn_disabled = (uploaded_file is None)
    analyze_btn = st.button("Run Analysis", type="primary", disabled=btn_disabled, width='stretch')

    if analyze_btn:

        if not question:
            st.warning("Please type a question first!")
        else:
            with st.spinner("Processing... "):
                try:
                    files = {"image": uploaded_file.getvalue()}
                    data = {"question": question}
                    
                    api_url = f"{URL}/predict"
                    response = requests.post(api_url, files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.markdown("### LLaMA 3.1 Answer")
                        if result.get('answer'):
                            st.success(result['answer'])
                        else:
                            st.warning("No answer generated.")

                        st.markdown("### BLIP Perception")
                        st.info(result['caption'])

                        st.markdown("### CLIP Confidence")
                        score = result['confidence_score']
                        st.progress(score)
                        
                        if score > 0.25:
                            st.write(f"**Score:** :green[{score:.3f}] (High)")
                        else:
                            st.write(f"**Score:** :red[{score:.3f}] (Low)")

                    else:
                        st.error("Server Error")

                except:
                    st.error("Connection Error!")