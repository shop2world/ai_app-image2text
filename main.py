from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import requests
import pyttsx3
import os
import streamlit as st

# .env 파일 로드
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")


# 이미지에서 텍스트 추출
def img2txt(path):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = pipe(path)[0]["generated_text"]
    print("추출된 텍스트:", text)
    return text


# 텍스트를 기반으로 이야기 생성
def story_making(context):
    template = """
    당신은 이야기꾼입니다.
    간단한 스토리에 기반한 짧은 이야기를 생성할 수 있습니다. 이야기는 30단어를 넘지 않아야 합니다.
    맥락: {context}
    이야기:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context"])
    story_llm = LLMChain(llm=ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0.8)
    , prompt=prompt, verbose=True)
    story = story_llm.predict(context=context, max_length=50)
    print("생성된 이야기:", story)
    return story

# 추출된 텍스트를 사용하여 story_making 호출
image_text = img2txt("llama2.png")
story = story_making(image_text)






# # 음성 생성
# def text2speech(input_text):
#     API_URL = "https://api-inference.huggingface.co/models/suno/bark-small"
#     headers = {"Authorization": f"Bearer {API_KEY}"}
#     payload = {
#         "inputs": input_text
#     }
#     response = requests.post(API_URL, headers=headers, json=payload)
        
#     with open('speech.flac', 'wb') as file:
#         file.write(response.content)

# # story를 사용하여 text2speech 호출
# text2speech(story)




# 음성 생성
def text2speech(input_text, output_path="speech.mp3"):
    # 음성으로 변환
    engine = pyttsx3.init()

    # 텍스트를 음성으로 변환
    engine.say(input_text)

    # 음성을 파일로 저장
    engine.save_to_file(input_text, output_path)

    # 변환된 음성을 재생
    engine.runAndWait()

# story를 사용하여 text2speech 호출
text2speech(story)

#스트림릿으로 이동text2speech(story)

def ui():
    st.header("이미지를 음성 이야기로")
    uploaded_file= st.file_uploader("이미지를 올려주세요..", type="png")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',use_column_width=True) 
        image_text = img2txt(uploaded_file.name)
        story = story_making(image_text)
        text2speech(story)
        with st.expander("image_text"):
            st.write(image_text)
        with st.expander("story"):
            st.write(story)    
        st.audio("speech.mp3")
if __name__ == '__main__':
    ui()
