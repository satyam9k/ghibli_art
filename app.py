import streamlit as st
import os
import mimetypes
import tempfile
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');

    * {
        font-family: 'Merriweather', serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #EB3678, #4F1787);
    }

    .css-18e3th9, .stTitle {
        font-size: 2.5rem;
        color: #fff;
        font-weight: bold;
    }

    .stRadio label {
        font-size: 1.2rem;
        color: #FFFFFF;
    }

    .stButton>button {
        background-color: #180161;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        transition: background-color 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4F1787;
    }

    .stFileUploader {
        background-color: #7E1891;
        border: 2px dashed #FFB6C1;
        border-radius: 10px;
        padding: 1.5rem;
    }

    .stAlert {
        font-size: 1rem;
        font-weight: bold;
    }
    
    .stDownloadButton>button {
        background-color: #FFB6C1;
        color: black;
        border-radius: 10px;
        font-size: 1rem;
        padding: 0.5rem 1rem;
    }

    .stSpinner {
        font-size: 1.2rem;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.title("👨🏻‍🎨✨👩🏻‍🎨")
mode = st.radio("Choose a transformation mode:", (
    "Ghibli Style Artwork", 
    "Classic Anime Style", 
    "Playing in Ghibli World"
))
if mode == "Ghibli Style Artwork":
    prompt = (
        "Transform the uploaded image into a Studio Ghibli style artwork while staying as true as possible "
        "to the original image. Focus on preserving the subject, facial features, expressions, and the background with high detail. "
        "Produce a cute and soft output with gentle lighting, vibrant colors, and hand-painted textures. "
        "Avoid hallucinating elements not present in the original image and maintain context fidelity."
    )
elif mode == "Classic Anime Style":
    prompt = (
        "Transform the uploaded image into a classic anime style artwork that harkens back to traditional hand-drawn techniques. "
        "Emphasize clean, expressive facial features and nostalgic, soft color palettes with dynamic yet gentle lighting. "
        "Ensure the style remains faithful to the original subject while producing a cute, soft, and timeless anime aesthetic without adding extraneous details."
    )
else:  # Playing in Ghibli World
    prompt = (
        "Reimagine the uploaded image as if the subject is playing in a magical Studio Ghibli world. "
        "Blend the subject seamlessly with an enchanting, vibrant Ghibli-inspired setting filled with whimsical details and lush natural scenery. "
        "Focus on preserving the subject's clear facial features and expressions while depicting them engaging in playful activity within a richly detailed, fantastical environment. "
        "Produce a cute and soft output with gentle lighting and harmonious colors, ensuring the subject remains central and true to the original image."
    )
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    if st.button("Transform Image"):
        with st.spinner("Transforming image..."):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_path = generate(temp_file_path, prompt)
            os.remove(temp_file_path)
            if output_path:
                st.success("Image transformed successfully!")
                st.image(output_path, caption="Transformed Image", use_container_width=True)
                with open(output_path, "rb") as file:
                    mime_type = "image/jpeg" if output_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    st.download_button(label="Download Transformed Image", data=file, file_name=output_path, mime=mime_type)
# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
#     if st.button("Transform Image"):
#         with st.spinner("Transforming image..."):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
#                 temp_file.write(uploaded_file.getbuffer())
#                 temp_file_path = temp_file.name

#             output_path = generate(temp_file_path, prompt)

#             # Ensure we delete temp file only if processing succeeded
#             if output_path:
#                 os.remove(temp_file_path)
#                 st.success("Image transformed successfully!")
#                 st.image(output_path, caption="Transformed Image", use_container_width=True)

#                 # Open transformed file and add download button
#                 with open(output_path, "rb") as file:
#                     mime_type = "image/jpeg" if output_path and output_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
#                     st.download_button(label="Download Transformed Image", data=file, file_name=os.path.basename(output_path), mime=mime_type)
#             else:
#                 st.error("❌ Image transformation failed. Please try again.")

# import streamlit as st
# import tempfile
# import os
# import uuid  # To generate unique filenames per user
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv
# load_dotenv()

# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');

#     * {
#         font-family: 'Merriweather', serif !important;
#     }
    
#     .stApp {
#         background: linear-gradient(135deg, #EB3678, #4F1787);
#     }

#     .css-18e3th9, .stTitle {
#         font-size: 2.5rem;
#         color: #fff;
#         font-weight: bold;
#     }

#     .stRadio label {
#         font-size: 1.2rem;
#         color: #FFFFFF;
#     }

#     .stButton>button {
#         background-color: #180161;
#         color: white;
#         border-radius: 10px;
#         border: none;
#         padding: 0.6rem 1.5rem;
#         font-size: 1rem;
#         transition: background-color 0.3s ease-in-out;
#     }
#     .stButton>button:hover {
#         background-color: #4F1787;
#     }

#     .stFileUploader {
#         background-color: #7E1891;
#         border: 2px dashed #FFB6C1;
#         border-radius: 10px;
#         padding: 1.5rem;
#     }

#     .stAlert {
#         font-size: 1rem;
#         font-weight: bold;
#     }
    
#     .stDownloadButton>button {
#         background-color: #FFB6C1;
#         color: black;
#         border-radius: 10px;
#         font-size: 1rem;
#         padding: 0.5rem 1rem;
#     }

#     .stSpinner {
#         font-size: 1.2rem;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# # Function to save binary file
# def save_binary_file(file_name, data):
#     with open(file_name, "wb") as f:
#         f.write(data)

# # Function to call Gemini API
# def generate(image_path, style_prompt):
#     client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
#     try:
#         files = [client.files.upload(file=image_path)]
#         st.write("✅ File uploaded successfully!")
#     except Exception as e:
#         st.error(f"❌ Error uploading file: {e}")
#         return None

#     model = "gemini-2.0-flash-exp-image-generation"
#     contents = [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
#                 types.Part.from_text(text=style_prompt)
#             ]
#         )
#     ]

#     generate_content_config = types.GenerateContentConfig(
#         response_modalities=["image"],
#         safety_settings=[types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF")],
#         response_mime_type="text/plain"
#     )

#     try:
#         for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
#             if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
#                 continue
#             if chunk.candidates[0].content.parts[0].inline_data:
#                 file_name = f"transformed_{uuid.uuid4().hex}.png"  # Unique filename for each user
#                 inline_data = chunk.candidates[0].content.parts[0].inline_data
#                 save_binary_file(file_name, inline_data.data)
#                 return file_name
#     except Exception as e:
#         st.error(f"❌ Error during image generation: {e}")
#         return None

# # UI Setup
# st.title("👨🏻‍🎨✨👩🏻‍🎨")

# mode = st.radio("Choose a transformation mode:", (
#     "Ghibli Style Artwork", 
#     "Classic Anime Style", 
#     "Playing in Ghibli World"
# ))

# if mode == "Ghibli Style Artwork":
#     prompt = (
#         "Transform the uploaded image into a Studio Ghibli style artwork while staying as true as possible "
#         "to the original image. Focus on preserving the subject, facial features, expressions, and the background with high detail. "
#         "Produce a cute and soft output with gentle lighting, vibrant colors, and hand-painted textures. "
#         "Avoid hallucinating elements not present in the original image and maintain context fidelity."
#     )
# elif mode == "Classic Anime Style":
#     prompt = (
#         "Transform the uploaded image into a classic anime style artwork that harkens back to traditional hand-drawn techniques. "
#         "Emphasize clean, expressive facial features and nostalgic, soft color palettes with dynamic yet gentle lighting. "
#         "Ensure the style remains faithful to the original subject while producing a cute, soft, and timeless anime aesthetic without adding extraneous details."
#     )
# else:  # Playing in Ghibli World
#     prompt = (
#         "Reimagine the uploaded image as if the subject is playing in a magical Studio Ghibli world. "
#         "Blend the subject seamlessly with an enchanting, vibrant Ghibli-inspired setting filled with whimsical details and lush natural scenery. "
#         "Focus on preserving the subject's clear facial features and expressions while depicting them engaging in playful activity within a richly detailed, fantastical environment. "
#         "Produce a cute and soft output with gentle lighting and harmonious colors, ensuring the subject remains central and true to the original image."
#     )

# # File upload
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#     if st.button("Transform Image"):
#         with st.spinner("Transforming image..."):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
#                 temp_file.write(uploaded_file.getbuffer())
#                 temp_file_path = temp_file.name

#             output_path = generate(temp_file_path, prompt)
#             os.remove(temp_file_path)  # Clean up temp file

#             if output_path:
#                 # Store file in session state for multi-user support
#                 if "output_image" not in st.session_state:
#                     st.session_state["output_image"] = {}

#                 user_id = str(uuid.uuid4())  # Unique ID for user session
#                 st.session_state["output_image"][user_id] = output_path

#                 st.success("Image transformed successfully!")
#                 st.image(output_path, caption="Transformed Image", use_container_width=True)

#                 # Download button
#                 with open(output_path, "rb") as file:
#                     mime_type = "image/jpeg" if output_path.endswith((".jpg", ".jpeg")) else "image/png"
#                     st.download_button(label="Download Transformed Image", data=file, file_name=os.path.basename(output_path), mime=mime_type)
