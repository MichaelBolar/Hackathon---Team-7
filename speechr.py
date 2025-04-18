# from gtts import gTTS
# import os

# text = "Hello, blah blah blah."

# tts_engine = gTTS(text)
# tts_engine.save("text.mp3")


# os.system("afplay text.mp3")

import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as mic:
    print("Listening")

    while (1):
        try:
            audio = recognizer.listen(mic) 
            recognized_text = recognizer.recognize_google(audio)
            print(recognized_text)
        except Exception:
            pass  # ignore everything and keep listening