from flask import Flask, request, render_template_string
from PIL import Image, ImageChops
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import base64
import io
import os

app = Flask(__name__)

# ── Load CNN model ──────────────────────────────
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(
    r"C:\Users\victus\OneDrive\New folder\forgery project\model.pth",
    map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── ELA Function ────────────────────────────────
def run_ela(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img.convert("RGB"), compressed)
    arr = np.array(diff)
    amplified = np.clip(arr * 15, 0, 255).astype(np.uint8)
    score = float(np.mean(amplified))
    ela_img = Image.fromarray(amplified)
    buf2 = io.BytesIO()
    ela_img.save(buf2, "PNG")
    ela_b64 = base64.b64encode(buf2.getvalue()).decode()
    return score, ela_b64

# ── HTML Page ───────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Document Forgery Detector</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .upload-box { background: white; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        input[type=file] { margin: 20px 0; }
        button { background: #4CAF50; color: white; padding: 12px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        button:hover { background: #45a049; }
        .result { margin-top: 30px; padding: 20px; border-radius: 10px; text-align: center; }
        .real { background: #d4edda; border: 2px solid #28a745; }
        .forged { background: #f8d7da; border: 2px solid #dc3545; }
        .images { display: flex; gap: 20px; justify-content: center; margin-top: 20px; }
        .images div { text-align: center; }
        img { max-width: 350px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .score { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .verdict { font-size: 32px; font-weight: bold; margin: 15px 0; }
    </style>
</head>
<body>
    <h1>🔍 Document Forgery Detector</h1>
    <div class="upload-box">
        <h3>Upload a document to check if it's genuine or forged</h3>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required><br>
            <button type="submit">Analyze Document</button>
        </form>
    </div>

    {% if result %}
    <div class="result {{ 'real' if result.verdict == 'GENUINE' else 'forged' }}">
        <div class="verdict">
            {{ '✅ GENUINE' if result.verdict == 'GENUINE' else '❌ FORGED' }}
        </div>
        <p><b>CNN Model:</b> {{ result.cnn_class }} — {{ result.cnn_conf }}% confident</p>
        <p><b>ELA Score:</b> {{ result.ela_score }} 
           {{ '(Clean ✅)' if result.ela_score < 15 else '(Suspicious ⚠️)' }}
        </p>
    </div>

    <div class="images">
        <div>
            <h3>Original Document</h3>
            <img src="data:image/png;base64,{{ result.original_b64 }}">
        </div>
        <div>
            <h3>ELA Analysis</h3>
            <img src="data:image/png;base64,{{ result.ela_b64 }}">
        </div>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        # CNN prediction
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            classes = ["forged", "real"]
            cnn_class = classes[prob.argmax()]
            cnn_conf = round(prob.max().item() * 100, 1)

        # ELA
        ela_score, ela_b64 = run_ela(img)
        ela_score = round(ela_score, 2)

        # Final verdict
        if cnn_class == "real" and ela_score < 15:
            verdict = "GENUINE"
        elif cnn_class == "forged" or ela_score > 20:
            verdict = "FORGED"
        else:
            verdict = "SUSPICIOUS"

        # Original image to base64
        buf = io.BytesIO()
        img.save(buf, "PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode()

        result = {
            "verdict": verdict,
            "cnn_class": cnn_class.upper(),
            "cnn_conf": cnn_conf,
            "ela_score": ela_score,
            "ela_b64": ela_b64,
            "original_b64": original_b64
        }

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    print("Starting server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True)