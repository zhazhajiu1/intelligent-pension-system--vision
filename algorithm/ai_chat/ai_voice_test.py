# from google.cloud import texttospeech
# import os
#
# # 设置 Google Cloud 的 API 密钥
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai-voice-428509-c4ee044c9c99.json"
#
# def speak(text):
#     client = texttospeech.TextToSpeechClient()
#
#     input_text = texttospeech.SynthesisInput(text=text)
#
#     # 设置声音参数
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="zh-CN",
#         name="zh-CN-Wavenet-A",  # 选择一个声音模型
#         ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
#     )
#
#     # 设置音频配置
#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=texttospeech.AudioEncoding.MP3
#     )
#
#     response = client.synthesize_speech(
#         input=input_text, voice=voice, audio_config=audio_config
#     )
#
#     # 保存生成的音频到文件
#     with open("output.mp3", "wb") as out:
#         out.write(response.audio_content)
#         print("写入文件'output.mp3'的音频内容")
#
#     # 播放音频
#     os.system("start output.mp3")
#
# # 使用新的 speak 函数
# speak("你好，这是一个个性化的语音示例。")

from gtts import gTTS
import os

text = "你好，世界,打爆小学期！我要放假"
output = gTTS(text=text, lang='zh-cn', slow=False)

output.save("output.mp3")

# 播放音频文件（需要安装pygame库）
import pygame

pygame.mixer.init()
pygame.mixer.music.load("output.mp3")
pygame.mixer.music.play()

