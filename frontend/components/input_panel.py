import streamlit as st

def render_input_panel():
    st.header("INPUT")
    input_type = st.radio("Input Type", ["Text", "Screenshot", "Reddit URL", "Image + Text"])
    
    user_input = None
    if input_type == "Text":
        user_input = st.text_area("Paste a social-media post or claim here...", height=150)
    elif input_type == "Screenshot":
        user_input = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
    elif input_type == "Reddit URL":
        user_input = st.text_input("Enter Reddit URL")
    
    return input_type, user_input
