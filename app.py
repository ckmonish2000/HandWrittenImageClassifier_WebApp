from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    print(request.json["hello"])
    return jsonify({"result": 3000})


if __name__ == "__main__":
    app.run(debug=True)