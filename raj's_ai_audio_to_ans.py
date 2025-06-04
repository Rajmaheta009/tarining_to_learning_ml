import threading
import speech_recognition as sr
import pyttsx3
import requests
from datetime import datetime

OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

r = sr.Recognizer()
stop_speaking_flag = threading.Event()

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate + 50)  # Increase speaking speed

STOP_WORDS = ("stop", "wait", "sorry", "what you said")

def speak(text):
    words = text.split()
    for word in words:
        if stop_speaking_flag.is_set():
            print("Speech interrupted!")
            engine.stop()
            break
        engine.say(word)
        engine.runAndWait()
    engine.stop()

def listen_for_interrupt():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening for 'stop' interrupt...")
        while not stop_speaking_flag.is_set():
            try:
                audio = r.listen(source, timeout=2, phrase_time_limit=2)
                interrupt_text = r.recognize_google(audio).lower()
                print("Interrupt heard:", interrupt_text)
                if any(word in interrupt_text for word in STOP_WORDS):
                    stop_speaking_flag.set()
            except Exception:
                # Timeout or unintelligible speech, continue listening
                pass

def record_text():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening for your input...")
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio).lower().strip()
            print("You said:", text)
            return text
        except Exception as e:
            print("No input or error:", e)
            return ""

def query_ollama(prompt, model=OLLAMA_MODEL):
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            print(f"Ollama Error: {response.status_code}")
            return "Sorry, I had trouble communicating with the model."
    except Exception as e:
        print(f"Request failed: {e}")
        return "Sorry, I had trouble communicating with the model."

def save_to_file(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("output.txt", "a") as f:
        f.write(f"[{timestamp}] {text}\n")

# === Main loop ===
while True:
    user_input = record_text()
    if not user_input:
        continue

    if user_input in STOP_WORDS:
        speak("Stopping the program.")
        break

    save_to_file(f"You: {user_input}")

    model_response = query_ollama(user_input)
    print("Model:", model_response)
    save_to_file(f"Model: {model_response}")

    stop_speaking_flag.clear()

    # Start listening for interrupt while speaking
    interrupt_thread = threading.Thread(target=listen_for_interrupt)
    interrupt_thread.start()

    speak(model_response)

    interrupt_thread.join()

    if stop_speaking_flag.is_set():
        speak("I heard you say stop. Please repeat.")
