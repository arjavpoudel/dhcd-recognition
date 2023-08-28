from flask import Flask, request, jsonify
from torch_utils import transform_image, get_prediction, get_char_pred


app = Flask(__name__)
ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        if not allowed_file(file.filename):
            return jsonify({"error": "file format not supported"})
        # try:
        image_bytes = file.read()
        tensor = transform_image(image_bytes)
        prediction = get_prediction(tensor)
        data = {
            "predicted char": get_char_pred(prediction),
            "class label": prediction.item(),
        }
        return jsonify(data)

        # except:
        #     return jsonify({"error": "error during prediction"})
    return jsonify({"result": 1})
