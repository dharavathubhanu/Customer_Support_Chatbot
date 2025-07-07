from flask import Flask, render_template, request, session
from chatbot_logic import chatbot_response
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # More secure random session key
chat_history = []

def log_to_file(user_msg, bot_msg):
    with open("chat_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"User: {user_msg}\n")
        log_file.write(f"Bot: {bot_msg}\n")
        log_file.write("-" * 40 + "\n")

@app.route("/")
def home():
    return render_template("index.html", history=chat_history)

@app.route("/get")
def get_bot_response():
    user_text = request.args.get("msg")
    previous_intent = session.get("last_intent")

    # Get response and current intent from logic
    bot_text, current_intent = chatbot_response(user_text, previous_intent)

    # Store chat history in memory
    chat_history.append({"user": user_text, "bot": bot_text})

    # Log to file
    log_to_file(user_text, bot_text)

    # Store intent for context-awareness
    session["last_intent"] = current_intent

    return bot_text

if __name__ == "__main__":
    print("\n✅ Server running at: http://127.0.0.1:5000/\n")
    app.run(debug=True)
