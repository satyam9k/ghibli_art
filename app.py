import streamlit as st
import os
import mimetypes
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def generate(image_path, style_prompt):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        files = [client.files.upload(file=image_path)]
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None
    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                types.Part.from_text(text=style_prompt)
            ]
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="")]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=["image", "text"],
        safety_settings=[types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF")],
        response_mime_type="text/plain"
    )
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            if chunk.candidates[0].content.parts[0].inline_data:
                file_name = "transformed_image"
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                output_file = f"{file_name}{file_extension}"
                save_binary_file(output_file, inline_data.data)
                st.write(f"File of mime type {inline_data.mime_type} saved to: {output_file}")
                return output_file
            else:
                st.write(chunk.text)
    except Exception as e:
        st.error(f"Error during image generation: {e}")
        return None

st.title("High Quality Style Image Transformer")
mode = st.radio("Choose a transformation mode:", (
    "High Quality Ghibli Style Artwork", 
    "Modern Day Anime Style", 
    "Playing in Ghibli World"
))
if mode == "High Quality Ghibli Style Artwork":
    prompt = (
        "Transform the uploaded image into a Studio Ghibli style artwork while staying as true as possible "
        "to the original image. Focus on preserving the subject, facial features, expressions, and the background with high detail. "
        "Produce a cute and soft output with soft lighting, vibrant colors, and hand-painted textures. "
        "Avoid hallucinating elements not present in the original image and maintain context fidelity."
    )
elif mode == "Modern Day Anime Style":
    prompt = (
        "Transform the uploaded image into a modern day anime style artwork while staying true to the original image. "
        "Focus on preserving the subject's facial features and expressions with crisp clarity, vibrant colors, and dynamic lighting. "
        "Utilize sleek, modern anime aesthetics to produce a cute and soft output that retains the original context without extraneous details."
    )
else:  # Playing in Ghibli World
    prompt = (
        "Reimagine the uploaded image as if the subject is playing in a Studio Ghibli world. "
        "Blend the subject seamlessly with a magical, vibrant Ghibli-inspired setting filled with whimsical details and lush natural elements. "
        "Focus on preserving the subject's facial features and expressions while portraying them engaging in playful activity within a richly detailed, enchanting environment. "
        "Ensure the output is cute, soft, and faithful to the original image without introducing any unrelated elements."
    )
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Transform Image"):
        with st.spinner("Transforming image..."):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            output_path = generate(temp_file_path, prompt)
            os.remove(temp_file_path)
            if output_path:
                st.success("Image transformed successfully!")
                st.image(output_path, caption="Transformed Image", use_column_width=True)
                with open(output_path, "rb") as file:
                    mime_type = "image/jpeg" if output_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    st.download_button(label="Download Transformed Image", data=file, file_name=output_path, mime=mime_type)
