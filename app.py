from flask import Flask, render_template, request, send_from_directory
import requests, os

app = Flask(__name__)

AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:5000/detect")
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "outputs"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return "No file uploaded", 400

        # Send to AI backend
        files = {"image": (file.filename, file.stream, file.mimetype)}
        resp = requests.post(AI_BACKEND_URL, files=files)

        if resp.status_code != 200:
            return f"AI backend error: {resp.text}", 500

        data = resp.json()
        image_file = os.path.basename(data["result_image"])  # only filename
        return render_template("index.html", result=data, image_file=image_file)

    return render_template("index.html", result=None, image_file=None)

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
