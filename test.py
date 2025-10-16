import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
from langdetect import detect
import os

# Load QA model
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")

# Hindi disaster awareness context
context_hi = """
बाढ़ के समय ऊँचाई वाली जगह पर जाएं। सड़क पर बहते पानी से दूर रहें। एक प्राथमिक उपचार किट, खाना और पीने का पानी पास रखें।
आग लगने पर तुरंत इमारत खाली करें और आपातकालीन सेवाओं को कॉल करें।
"""

# Speech-to-text in Hindi
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ बोलिए (Hindi/English)...")
        audio = recognizer.listen(source)

    try:
        # Set language to Hindi for Google recognizer
        query = recognizer.recognize_google(audio, language='hi-IN')
        print(f"📝 आपने कहा: {query}")
        return query
    except sr.UnknownValueError:
        print("❌ पहचान नहीं हो सकी।")
        return None

# Text-to-speech
def speak_answer(answer):
    lang = detect(answer)
    tts = gTTS(answer, lang=lang)
    tts.save("response.mp3")
    os.system("afplay response.mp3")  # ✅ Mac

# Full pipeline
def run_bot():
    question = get_voice_input()
    if question:
        result = qa_pipeline(question=question, context=context_hi)
        answer = result['answer']
        print(f"🤖 AIDAR का उत्तर: {answer}")
        speak_answer(answer)

# Run the assistant
run_bot()
