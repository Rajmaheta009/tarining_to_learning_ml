from gtts import gTTS

# Create audio
tts = gTTS("Hello, how are you?")
tts.save("hello.wav")
