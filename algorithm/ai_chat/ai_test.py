import pyttsx3

# 初始化语音引擎
engine = pyttsx3.init()

# 列出所有可用的声音
voices = engine.getProperty('voices')
for idx, voice in enumerate(voices):
    print(f"Voice {idx}:")
    print(f" - ID: {voice.id}")
    print(f" - Name: {voice.name}")
    print(f" - Languages: {voice.languages}")
    print(f" - Gender: {voice.gender}")
    print(f" - Age: {voice.age}")

# 选择一个声音，例如选择第一个声音（Microsoft Huihui Desktop - Chinese (Simplified)）
engine.setProperty('voice', voices[0].id)

# 设置语速和音量
engine.setProperty('rate', 150)  # 语速
engine.setProperty('volume', 1.0)  # 音量

# 进行文本到语音的转换
def speak(text):
    engine.say(text)
    engine.runAndWait()

# 示例文本
speak("你好，这是一个语音合成示例。")

# 如果需要测试不同的语音，可以再次设置并转换文本
engine.setProperty('voice', voices[1].id)
speak("Hello, this is a speech synthesis example.")
