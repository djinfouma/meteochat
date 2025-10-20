from flask import Flask, render_template, request, redirect, url_for, jsonify
from chatbot_expert import expert_chat, download_expert_report
from chatbot_standard import standard_chat, download_standard_report

app = Flask(__name__)

# Home page (mode selection)
@app.route('/')
def pre_index():
    return render_template("pre_index.html")

# Page Expert
@app.route('/expert')
def expert():
    return render_template("expert.html")

# Page Standard
@app.route('/standard')
def standard():
    return render_template("standard.html")

# API for chat expert
@app.route('/chat_expert', methods=['POST'])
def chat_expert():
    user_input = request.json.get("message")
    response = expert_chat(user_input)
    return jsonify({"response": response})

# API for chat standard
@app.route('/chat_standard', methods=['POST'])
def chat_standard():
    user_input = request.json.get("message")
    response = standard_chat(user_input)
    return jsonify({"response": response})

# Download report expert
@app.route('/download_expert')
def download_expert():
    return download_expert_report()

# Download report standard
@app.route('/download_standard')
def download_standard():
    return download_standard_report()

if __name__ == '__main__':
    app.run(debug=True)
