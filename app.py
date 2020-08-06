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


if __name__ == "__main__":
    app.run(debug=True)