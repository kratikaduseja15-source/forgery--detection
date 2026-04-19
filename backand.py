from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageChops
import torch
import torch.nn as nn
import numpy as np
import io
import base64
from torchvision import transforms, models

# ── Setup ────────────────────────────────────────
app = FastAPI(title="Document Forgery Detection API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ───────────────────────────────────
MODEL_PATH = r"C:\Users\victus\OneDrive\New folder\forgery project\model.pth"

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── ELA Function ─────────────────────────────────
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

# ── Routes ───────────────────────────────────────

# Home route — check if API is working
@app.get("/")
def home():
    return {
        "message": "Document Forgery Detection API",
        "status": "running ✅",
        "endpoints": [
            "/analyze  — POST — Upload document",
            "/health   — GET  — Check server",
            "/docs     — GET  — API documentation"
        ]
    }

# Health check
@app.get("/health")
def health():
    return {"status": "healthy ✅"}

# Main analyze endpoint
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):

    # Check file type
    if not file.filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files allowed!"}
        )

    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # 1. CNN Prediction
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        classes = ["forged", "real"]
        cnn_class = classes[prob.argmax()]
        cnn_conf = round(prob.max().item() * 100, 1)

    # 2. ELA Analysis
    ela_score, ela_b64 = run_ela(img)
    ela_score = round(ela_score, 2)

    # 3. Final Verdict
    if cnn_class == "forged" or ela_score > 20:
        verdict = "FORGED"
        risk = "HIGH"
    elif ela_score > 15:
        verdict = "SUSPICIOUS"
        risk = "MEDIUM"
    else:
        verdict = "GENUINE"
        risk = "LOW"

    # 4. Reasons
    reasons = []
    if cnn_class == "forged":
        reasons.append(
            f"CNN detected forgery patterns ({cnn_conf}%)")
    if ela_score > 15:
        reasons.append(
            f"ELA score {ela_score} exceeds safe limit of 15")
    if not reasons:
        reasons.append("No suspicious patterns detected")

    # 5. Original image to base64
    buf = io.BytesIO()
    img.save(buf, "PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode()

    # Return full response
    return {
        "filename": file.filename,
        "verdict": verdict,
        "risk_level": risk,
        "cnn": {
            "prediction": cnn_class.upper(),
            "confidence": cnn_conf
        },
        "ela": {
            "score": ela_score,
            "status": "suspicious" if ela_score > 15 else "clean",
            "image_base64": ela_b64
        },
        "reasons": reasons,
        "original_image_base64": original_b64
    }