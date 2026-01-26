import mlflow
import torch
import numpy as np
import cv2
import io
import base64
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from PIL import Image
from model_arch import NestedUNet
from mlflow.tracking import MlflowClient
import os

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DAGSHUB_REPO = "sam2800ml/fluorescence-segmentation"
RUN_ID = "run_id_env"
ARTIFACT_PATH = "model"           # folder
STATE_DICT_NAME = "model.pth"     # file inside folder

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.model = None

    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO}.mlflow")
    device = torch.device("cpu")
    app.state.device = device

    try:
        print("📦 Downloading model artifacts from DagsHub...")

        client = MlflowClient(
            tracking_uri=f"https://dagshub.com/{DAGSHUB_REPO}.mlflow"
        )

        local_dir = client.download_artifacts(
            run_id=RUN_ID,
            path=ARTIFACT_PATH
        )

        state_dict_path = os.path.join(local_dir, STATE_DICT_NAME)

        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Missing {STATE_DICT_NAME} in artifacts")

        print("🧠 Initializing model architecture...")
        model = NestedUNet(in_channels=3, num_classes=4)

        print("📥 Loading state_dict only (low RAM)...")
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)

        del state_dict
        gc.collect()

        model.to(device)
        model.eval()

        app.state.model = model
        app.state.ready = True

        print("✅ Model loaded safely and ready for inference.")

    except Exception as e:
        print(f"❌ Model load failed: {e}")
        app.state.model = None
        app.state.ready = False

    yield

    if app.state.model is not None:
        del app.state.model
        gc.collect()

    print("🧹 Backend shutdown complete.")

app = FastAPI(lifespan=lifespan)

def preprocess(image_bytes: bytes, device):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (512, 512))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_norm = (img_resized / 255.0 - mean) / std

    tensor = (
        torch.from_numpy(img_norm)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    return tensor


@app.get("/health")
def health(request: Request):
    return {"ready": getattr(request.app.state, "ready", False)}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is still loading.")

    model = request.app.state.model
    device = request.app.state.device

    try:
        contents = await file.read()
        input_tensor = preprocess(contents, device)

        with torch.inference_mode():
            output = model(input_tensor)
            mask = (
                torch.argmax(output, dim=1)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        success, buffer = cv2.imencode(".png", mask)
        if not success:
            raise RuntimeError("PNG encoding failed")

        mask_base64 = base64.b64encode(buffer).decode("utf-8")

        unique, counts = np.unique(mask, return_counts=True)
        dist = {str(k): int(v) for k, v in zip(unique, counts)}

        del input_tensor, output, mask
        gc.collect()

        return {
            "mask_base64": mask_base64,
            "class_distribution": dist,
            "status": "success"
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
