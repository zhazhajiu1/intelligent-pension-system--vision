import speech_recognition as sr  # 捕获麦克风输入并将其转换为文本
import pyttsx3
import openai
import requests

# 配置OpenAI API
openai.api_key = 'Bearer sk-ebn5Il2cyqaOEohGh6jbKSmODl2hLi0gwyp5ihJLE8QJtzjT'

# 存储用户上下文
context = [{"role": "user", "content": "我想咨询一些问题，问题应用场景为养老系统，请你简短回答我后续的问题,不要说多余的话"},
           {"role": "assistant", "content": "好的，说出您的问题即可，我会用简短的语言试着帮您解答~"}]

# 初始化语音识别和语音合成引擎
recognizer = sr.Recognizer()
engine = pyttsx3.init()  # 文本回应转换为语音输出

# 语音合成函数
def speak(text):
    # 文本到语音的转换
    engine.say(text)
    engine.runAndWait()

# 语音识别函数
def recognize_speech():
    with sr.Microphone() as source:
        print("请开始说话...")
        speak("您好，请问您有什么问题...若要退出语音聊天功能,请说退出 或者 停止 或者 拜拜")
        audio = recognizer.listen(source)
        try:
            # 使用 Google 的语音识别 API，将音频转换为文本
            text = recognizer.recognize_google(audio, language="zh-CN")
            print("你说: " + text)
            # speak(text)
            return text
        except sr.UnknownValueError:
            print("无法识别音频")
            speak("抱歉，无法识别音频")
            return None
        except sr.RequestError as e:
            print(f"请求错误: {e}")
            speak("抱歉，请求错误")
            return None


# def chat(user_input):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=user_input,
#         max_tokens=150
#     )
#     message = response.choices[0].text.strip()
#     return message

# 调用OpenAI GPT-3进行对话
def chat_with_ai(content):

    # print('接受内容',content)

    # API 地址
    url = "https://api.chatanywhere.tech/v1/chat/completions"

    user_message = content

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": context + [{"role": "user", "content": user_message}],
        "temperature": 0.7
    }

    # 请求头部，包括内容类型和授权信息
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-3pBAXdmPWDdruZqaiM7VaGQKMhPg4VYcaITHImVgGDeIflsr"
        "Authorization": "Bearer sk-ebn5Il2cyqaOEohGh6jbKSmODl2hLi0gwyp5ihJLE8QJtzjT"
    }

    try:
        # 发送 POST 请求
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 将触发异常，如果HTTP请求返回非200状态码
        message = response.json()
        reply = message['choices'][0]['message']['content']
    except requests.RequestException as e:
        # 处理HTTP请求相关的异常
        print(f"Request failed: {e}")
        return "Internal Server Error", 500
    except KeyError as e:

        print(f"报错: {e}")
        return "Bad response from AI API", 500

    # reply = reply[10:]

    print("机器人: ",reply)
    context.append({"role": "user", "content": user_message})
    context.append({"role": "assistant", "content": reply})

    return reply

# 主函数
def main():
    while True:
        user_input = recognize_speech()
        if user_input:
            if user_input.lower() in ["退出", "停止", "拜拜"]:
                speak("再见")
                break
            # response = chat(user_input)
            response = chat_with_ai(user_input)
            speak(response)

if __name__ == "__main__":
    main()
