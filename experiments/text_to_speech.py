"""from gtts import gTTS
import os

def text_to_speech(text, lang, filename):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    os.system(f"start {filename}")

text = "Hello, I am converted text to speech in Python."
lang = 'en'
filename = "output.mp3"
text_to_speech(text, lang, filename)"""

# Import the required module for text 
# to speech conversion
import pyttsx3

# init function to get an engine instance for the speech synthesis 
engine = pyttsx3.init()

# say method on the engine that passing input text to be spoken
engine.say('Hello sir, how may I help you, sir.')

# run and wait method, it processes the voice commands. 
engine.runAndWait()

