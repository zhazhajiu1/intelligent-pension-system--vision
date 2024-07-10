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

