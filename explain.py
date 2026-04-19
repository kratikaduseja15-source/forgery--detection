import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFont
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64

# ── Load Model ──────────────────────────────────
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

# ── ELA ─────────────────────────────────────────
def run_ela(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img.convert("RGB"), compressed)
    arr = np.array(diff)
    amplified = np.clip(arr * 15, 0, 255).astype(np.uint8)
    score = float(np.mean(amplified))
    return score, Image.fromarray(amplified)

# ── GradCAM (highlights suspicious regions) ─────
def get_gradcam(img_pil):
    tensor = transform(img_pil).unsqueeze(0)
    tensor.requires_grad_(True)

    # Get last conv layer
    features = None
    grads = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_backward_hook(backward_hook)

    output = model(tensor)
    pred = output.argmax()
    output[0, pred].backward()

    handle_f.remove()
    handle_b.remove()

    # Generate heatmap
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * features).sum(dim=1).squeeze()
    cam = torch.relu(cam).detach().numpy()
    cam = cv2.resize(cam, (img_pil.width, img_pil.height))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# ── Generate Full Report ─────────────────────────
def generate_report(image_path):
    print(f"\n🔍 Generating Explainability Report for:")
    print(f"   {image_path}\n")

    img = Image.open(image_path).convert("RGB")

    # 1. CNN Prediction
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        classes = ["forged", "real"]
        cnn_class = classes[prob.argmax()]
        cnn_conf = round(prob.max().item() * 100, 1)

    # 2. ELA
    ela_score, ela_img = run_ela(img)
    ela_score = round(ela_score, 2)

    # 3. GradCAM
    cam = get_gradcam(img)

    # 4. Create heatmap overlay
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_array = np.array(img)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    # 5. Final verdict
    if cnn_class == "forged" or ela_score > 20:
        verdict = "FORGED ❌"
        risk = "HIGH"
    elif ela_score > 15:
        verdict = "SUSPICIOUS ⚠️"
        risk = "MEDIUM"
    else:
        verdict = "GENUINE ✅"
        risk = "LOW"

    # 6. Generate report figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Document Forgery Analysis Report\n"
        f"Verdict: {verdict}  |  Risk: {risk}",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(img)
    axes[0].set_title("Original Document")
    axes[0].axis("off")

    axes[1].imshow(ela_img)
    axes[1].set_title(
        f"ELA Analysis\nScore: {ela_score} "
        f"({'Suspicious' if ela_score > 15 else 'Clean'})"
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"AI Attention Map\nCNN: {cnn_class.upper()} "
        f"({cnn_conf}%)"
    )
    axes[2].axis("off")

    # 7. Add explanation text
    reasons = []
    if cnn_class == "forged":
        reasons.append(f"• CNN model flagged as FORGED ({cnn_conf}% confident)")
    if ela_score > 15:
        reasons.append(f"• ELA score {ela_score} exceeds threshold (>15 = suspicious)")
    if ela_score > 20:
        reasons.append(f"• High ELA score indicates image manipulation")
    if not reasons:
        reasons.append("• No suspicious patterns detected")

    reason_text = "\n".join(reasons)
    fig.text(0.5, 0.02, f"Reasons: {reason_text}",
             ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("explainability_report.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # 8. Print report
    print("=" * 50)
    print("    DOCUMENT FORGERY ANALYSIS REPORT")
    print("=" * 50)
    print(f"Verdict:      {verdict}")
    print(f"Risk Level:   {risk}")
    print(f"CNN Model:    {cnn_class.upper()} ({cnn_conf}%)")
    print(f"ELA Score:    {ela_score}")
    print(f"\nReasons:")
    for r in reasons:
        print(f"  {r}")
    print("=" * 50)
    print("Report saved: explainability_report.png ✅")


# Run it
IMAGE = r"C:\Users\victus\OneDrive\New folder\forgery project\data\findit2\test\X00016469619.png"
generate_report(IMAGE)