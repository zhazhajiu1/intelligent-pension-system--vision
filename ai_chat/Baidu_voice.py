import re
import sys, os
import pyttsx3
import win32com.client as wincl
import chardet
from pathlib import Path


def audition():
    # 创建对象
    engine = pyttsx3.init()

    # 获取当前语音速率
    rate = engine.getProperty('rate')
    # print(f'语音速率：{rate}')
    # 设置新的语音速率
    engine.setProperty('rate', 150)

    # 获取当前语音音量
    volume = engine.getProperty('volume')
    # print(f'语音音量：{volume}')
    # 设置新的语音音量，音量最小为 0，最大为 1
    engine.setProperty('volume', 1.0)

    # 获取当前语音声音的详细信息，并试听语音
    voices = engine.getProperty('voices')

    # 從語音信息是提取機器人姓名並試音
    v = 0  # 语音索引号
    for voice in voices:
        # print(f'语音声音详细信息：{voice}')
        # print(type(voice))
        # str1 = str(voice)[82:100]
        str1 = str(voice)[100:140].lstrip()  # 將多行語音信息提取部分並去掉左空格
        # print(v, '号', str1)
        # print(type(str1))
        pattern = r" (.*?) "  # 匹配規則：匹配兩者之间的内容
        str2 = re.search(pattern, str1).group(1)
        str3 = str(voice)[50:100].lstrip()
        if 'CN' in str3:
            str3 = '普通话'
            str4 = '我会说'
        if 'HK' in str3:
            str3 = '粤语'
            str4 = '我会说'
        if 'TW' in str3:
            str3 = '国语'
            str4 = '我会说'
        if 'EN' in str3:
            str3 = 'English'
            str4 = 'I am the ', v, 'voice robot, and my name is ', str2, 'I can speak'

        print(v, '号', str2, str3)
        engine.setProperty('voice', voices[v].id)
        engine.say("大家好，我是%d号语音机器人%s，%s%s" % (v, str2, str4, str3))
        engine.runAndWait()
        v = v + 1


audition()