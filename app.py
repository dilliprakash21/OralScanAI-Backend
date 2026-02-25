import os
import io
import time
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from anthropic import Anthropic

# =====================================================
# 1️⃣ FLASK SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# 2️⃣ CONFIG
# =====================================================
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Users\vdill\Downloads\oralscan_classification.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

API_KEY = "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"

client = Anthropic(api_key=API_KEY)

# =====================================================
# 3️⃣ MODEL
# =====================================================
class OralScanClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_plaque = nn.Linear(in_features, num_classes)
        self.fc_gingival = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        plaque_out = self.fc_plaque(features)
        gingival_out = self.fc_gingival(features)
        return plaque_out, gingival_out


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = OralScanClassifier()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded")
    return model


MODEL = load_model(MODEL_PATH)

# =====================================================
# 4️⃣ TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# 5️⃣ LABELS
# =====================================================
CLASS_LABELS = ["Low", "Moderate", "High", "Severe"]

# Clinical PI/GI mapping (0–3 scale)
INDEX_MAPPING = {
    0: 0.3,
    1: 1.2,
    2: 2.1,
    3: 2.8
}

# =====================================================
# 6️⃣ UTILITIES
# =====================================================
def calculate_confidence(pred_tensor, class_idx):
    prob = F.softmax(pred_tensor, dim=1)[0, class_idx]
    return float(prob * 100)


def class_to_index(class_idx):
    return INDEX_MAPPING.get(class_idx, 0.0)

# =====================================================
# 7️⃣ CLAUDE TEXT
# =====================================================
def generate_claude_text(plaqueIndex, giIndex, risk, confidence, time_taken):
    prompt = f"""
Return STRICT JSON only with keys:
plaqueText, gingivalText, recommendation.

Values:
PI={plaqueIndex}
GI={giIndex}
Risk={risk}
Confidence={confidence}
Time={time_taken}
"""

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()
        return json.loads(text)

    except Exception as e:
        print("Claude error:", e)
        return {
            "plaqueText": "Explanation unavailable.",
            "gingivalText": "Explanation unavailable.",
            "recommendation": "Consult a dental professional."
        }

# =====================================================
# 8️⃣ ROUTES
# =====================================================
@app.route("/")
def home():
    return "OralScan API running"


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    # ===== Model Prediction =====
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        plaque_pred, gingival_pred = MODEL(img_tensor)

        plaque_idx = int(torch.argmax(plaque_pred, dim=1).item())
        gingival_idx = int(torch.argmax(gingival_pred, dim=1).item())

        plaque_conf = calculate_confidence(plaque_pred, plaque_idx)
        gingival_conf = calculate_confidence(gingival_pred, gingival_idx)

    # Overall risk = worst
    overall_idx = max(plaque_idx, gingival_idx)
    risk_level = CLASS_LABELS[overall_idx]

    # ===== Clinical Scores =====
    plaqueIndex = class_to_index(plaque_idx)
    gingivalIndex = class_to_index(gingival_idx)

    screening_time = round(time.time() - start_time, 2)
    avg_conf = round((plaque_conf + gingival_conf) / 2, 1)

    # ===== Claude =====
    claude_data = generate_claude_text(
        plaqueIndex,
        gingivalIndex,
        risk_level,
        avg_conf,
        screening_time
    )

    # ===== FINAL RESPONSE =====
    return jsonify({
        # Required by frontend
        "plaqueIndex": round(plaqueIndex, 2),
        "gingivalIndex": round(gingivalIndex, 2),
        "riskLevel": risk_level,
        "aiConfidence": avg_conf,

        # Extra details
        "plaqueClass": CLASS_LABELS[plaque_idx],
        "gingivalClass": CLASS_LABELS[gingival_idx],
        "plaqueConfidence": round(plaque_conf, 1),
        "gingivalConfidence": round(gingival_conf, 1),
        "screeningTime": screening_time,

        **claude_data
    })


# =====================================================
# 9️⃣ RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)