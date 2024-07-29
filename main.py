import streamlit as st
import pypdf
from diffusers import StableDiffusionPipeline
import torch
import cv2 as cv
import numpy
import textwrap
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from IPython.display import Markdown


def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key


def sidebar():
    with st.sidebar:
        st.markdown("<h1 style='font-style: italic; color: #53C8E6;'>Hey <span style = 'color: #FFFFFF;'> There!</span></h1>", 
                    unsafe_allow_html = True)
        st.write("")

        st.subheader("ABOUT:")
        st.markdown("CoverCraftAI is a tool that brings life to your story, "
                    " allowing you to upload your stories and receive quick and "
                    "artistic image for your story, all powered by StableDiffusion Model & Google Gemini's advanced technology!")
        st.write("")
        st.write("")
        api_key_input = st.text_input(
            "Enter your Gemini API Key",
            type = "password",
            placeholder = "Paste your Gemini API key here (sk-...)",
            help = "You can get your API key from https://aistudio.google.com/app/u/1/apikey.",
            value = st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.subheader("HOW TO USE: ")
        st.markdown("<p style = 'cursor: default;'>1. Enter your Gemini's API KEY."
                    "<br>2. Upload your story file through the upload button."
                    "<br>3. Choose the artistic style that you prefer."
                    "<br>4. Click 'Generate Image'."
                    "<br>5. Relax while we generate your image."
                    "<br>6. Hurray! Your image's here!!", unsafe_allow_html = True)

        return api_key_input


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate = lambda _: True))


def generate_image(user_input, genre, api_key_input):
    GOOGLE_API_KEY = api_key_input
    genai.configure(api_key = GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Story: {user_input} Give me a descriptive prompt about this story in 1 line for image generation in stable diffusion model.")

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype = torch.float16)
    pipe = pipe.to("cuda")

    data = response.text
    prompt = f"Create an {genre} image about {data}"
    image = pipe(prompt)

    open_cv_image = numpy.array(image.images)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    color_coverted = cv.cvtColor(open_cv_image[0], cv.COLOR_BGR2RGB)

    pil_image = Image.fromarray(color_coverted)
    st.image(pil_image, caption = 'Image', use_column_width = True)

    buf = BytesIO()
    pil_image.save(buf, format =" JPEG")
    byte_im = buf.getvalue()

    btn = st.download_button(label = "Download image", data = byte_im, file_name = "image.jpeg", mime = "image/jpeg")


def read_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for i in range(num_pages):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    return text


def main():
    st.set_page_config(page_title = "CoverCraftAI・Streamlit", page_icon = "📖")
    api_key_input = sidebar()
    st.markdown("<h1 style = 'margin-bottom:-3%;'>Story<span style = 'color: #53C8E6;'> Vision</span></h1>", 
                unsafe_allow_html = True)
    st.markdown("<p style = 'padding-bottom: 2%'>📖 Crafting Visual Narratives</p>", unsafe_allow_html = True)

    genre = st.radio("Choose style", ["Pop Art", "Pointillism", "Impressionism", "Minimalism", "Cubism"])

    uploaded_file = st.file_uploader("Upload a PDF file", type = ["pdf"], help = "Scanned Documents aren't supported yet!")

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        text = read_pdf(uploaded_file)
        st.empty()
        st.markdown("---")

    if uploaded_file is not None:
        if st.button("Generate Image"):
            with st.spinner("Generating image... This may take a while ⏳"):
                generate_image(text, genre, api_key_input)

if __name__ == "__main__":
    main()
