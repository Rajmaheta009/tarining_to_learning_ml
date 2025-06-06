import threading
import speech_recognition as sr
import pyttsx3
import requests
import time
from datetime import datetime
import difflib

OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

r = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

STOP_WORDS = ("stop", "wait", "sorry", "what you said")

# Event to signal TTS speaking should stop
stop_speaking_flag = threading.Event()

# Flag to skip next input after interrupt
skip_next_input = False


def is_interrupt_command(text):
    text = text.lower().strip()
    for word in STOP_WORDS:
        ratio = difflib.SequenceMatcher(None, text, word).ratio()
        if ratio > 0.8:
            return True
    return False


def listen_for_interrupt():
    """Listen for interrupt commands only while speaking."""
    print("Interrupt listener started.")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.3)
        while not stop_speaking_flag.is_set():
            try:
                audio = r.listen(source, timeout=2, phrase_time_limit=2)
                interrupt_text = r.recognize_google(audio).lower().strip()
                print(f"Interrupt heard: '{interrupt_text}'")
                if is_interrupt_command(interrupt_text):
                    print("Interrupt command detected! Stopping speech.")
                    stop_speaking_flag.set()
                    # Stop pyttsx3 speaking immediately
                    engine.stop()
                    break
            except sr.WaitTimeoutError:
                # No speech detected within timeout
                continue
            except sr.UnknownValueError:
                # Speech not recognized
                continue
            except Exception as e:
                print(f"Interrupt listener error: {e}")
                continue
    print("Interrupt listener stopped.")


def speak(text):
    """Speak text but allow interrupt."""
    if stop_speaking_flag.is_set():
        return

    stop_speaking_flag.clear()

    # Start interrupt listener thread
    interrupt_thread = threading.Thread(target=listen_for_interrupt)
    interrupt_thread.daemon = True
    interrupt_thread.start()

    # Speak text (blocking call)
    engine.say(text)
    engine.runAndWait()

    # Once speaking is done, signal listener to stop
    stop_speaking_flag.set()
    interrupt_thread.join()


def record_text():
    """Capture user voice input."""
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.3)
        print("Listening for your input...")

        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio).lower().strip()
            print(f"You said: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I did not understand.")
            return ""
        except Exception as e:
            print(f"Error during recognition: {e}")
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


def main():
    global skip_next_input

    while True:
        if skip_next_input:
            print("Skipping next input due to interrupt.")
            skip_next_input = False
            # Flush microphone input buffer
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                try:
                    _ = r.listen(source, timeout=1, phrase_time_limit=1)
                except:
                    pass
            time.sleep(0.5)
            continue

        user_input = record_text()
        if not user_input:
            continue

        if is_interrupt_command(user_input):
            speak("Stopping the program.")
            break

        save_to_file(f"You: {user_input}")

        model_response = query_ollama(user_input)
        print(f"Model: {model_response}")
        save_to_file(f"Model: {model_response}")

        stop_speaking_flag.clear()
        speak(model_response)

        if stop_speaking_flag.is_set():
            speak("I heard you say stop. Please repeat.")
            skip_next_input = True
            continue

        # Small pause before next listening
        time.sleep(0.5)


if __name__ == "__main__":
    main()
