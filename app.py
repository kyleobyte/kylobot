from flask import Flask, render_template, request, jsonify
from kylobot import handle_message

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/message', methods=['POST'])
def message():
    user_input = request.form['message']
    assistant_response = handle_message(user_input)
    print(f"Assistant Response: {assistant_response}")
    return jsonify({"response": assistant_response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8727)
