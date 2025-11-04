from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import torch
from torchvision import transforms as T
import cv2
import numpy as np
from charm import charm, EnsembleModel

app = Flask(__name__)
img_size = 384
_COL_NORM = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

def tensor_to_base64(t: torch.Tensor):
    # convert from [C,H,W] to [H,W,C]
    t = t.detach().cpu().clamp(0, 1)
    img = T.ToPILImage()(t)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EnsembleModel(pretrained=False)
model.load_state_dict(torch.load("model/model.pth", map_location="cpu", weights_only=True))
model.eval()

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No  image file provided"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img = np.asarray(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
    t = _COL_NORM(t)
    t = t.unsqueeze(0)
    
    out = charm(t, device, model)

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
