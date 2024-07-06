from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

context = [{"role": "user", "content": "我想咨询一些问题，问题应用场景为养老系统，请你简短回答我后续的问题,不要说多余的话"},
           {"role": "assistant", "content": "好的，说出您的问题即可，我会用简短的语言试着帮您解答~"}]

@app.route('/chat', methods=['POST'])

def chat():
    data = request.json
    content = data['content']

    # API 地址
    url = "https://api.chatanywhere.tech/v1/chat/completions"

    user_message = content

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": context + [{"role": "user", "content": user_message}],
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-ebn5Il2cyqaOEohGh6jbKSmODl2hLi0gwyp5ihJLE8QJtzjT"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        message = response.json()
        reply = message['choices'][0]['message']['content']
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except KeyError as e:
        return jsonify({"error": str(e)}), 500

    context.append({"role": "user", "content": user_message})
    context.append({"role": "assistant", "content": reply})

    return jsonify(reply)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
