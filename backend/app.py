from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
    