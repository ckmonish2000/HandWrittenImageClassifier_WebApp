from flask import Flask, jsonify, request, render_template
from PIL import Image
import io
import torch
from torch_util import transformer, predicts
app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        file = request.files.get("test")
        file = file.read()
        x = transformer(file)
        x = predicts(x)
        print(x.item())
        return jsonify({"result": x.item()})
    return render_template("index.htm")


@app.route("/predict2", methods=["GET", "POST"])
def test2():
    if request.method == "POST":
        file = request.files.get("test")
        file = file.read()
        x = transformer(file)
        x = predicts(x)
        print(x.item())
        return render_template("index2.htm",
                               pred=f"{x.item()} is the number in the image")
    return render_template("index2.htm", pred="select a image for prediciton")


if __name__ == "__main__":
    app.run(debug=True)