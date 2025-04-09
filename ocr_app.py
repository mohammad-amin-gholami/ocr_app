import streamlit as st
from PIL import Image
import pytesseract
from googletrans import Translator
from gtts import gTTS
import os


st.title("OCR استخراج")

uploaded_file = st.file_uploader("یک تصویر آپلود کن", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='تصویر آپلود شده', use_container_width=True)

# انتخاب زبان
    lang_choice = st.selectbox("زبان متن داخل تصویر:", ["فارسی", "انگلیسی"])
    lang_code = 'fas' if lang_choice == "فارسی" else 'eng'
# استخراج متن با توجه به زبان انتخاب شده 
    if st.button("استخراج متن"):
        text = pytesseract.image_to_string(image, lang=lang_code)
        st.subheader("متن استخراج‌شده:")
        st.text_area("", text, height=200)

# ترجمه متن به زبان های دیگر
        # translator = Translator()
        # dest_lang = 'en' if lang_code == 'fas' else 'fa'
        # translated = translator.translate(text, src=lang_code, dest=dest_lang)
        # st.subheader("ترجمه:")
        # translated = asyncio.run(translate_text(translated))
        # st.text_area("", translated.text, height=200)

# تبدیل به فایل صوتی 
        # tts_lang = 'fa' if dest_lang == 'fa' else 'en'
        # tts = gTTS(translated.text, lang=tts_lang)
        # tts.save("output.mp3")
        # audio_file = open("output.mp3", "rb")
        # st.audio(audio_file.read(), format="audio/mp3")


        # os.remove("output.mp3")
