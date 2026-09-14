import os
import gc
import torch
from pathlib import Path
from ultralytics import YOLO

def train_stable():
    # 1. Clean VRAM Cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = 0
        print(f"[SYSTEM] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[WARN] Using CPU")

    # 2. Locate data.yaml
    base_dir = Path(__file__).resolve().parent.parent
    yaml_path = base_dir / "data" / "dataset" / "data.yaml"
    if not yaml_path.exists():
        yaml_path = base_dir / "data" / "data.yaml"

    if not yaml_path.exists():
        print(f"[ERROR] Could not find data.yaml at {yaml_path}")
        return

    # 3. Use a distinct run name for the 2-class CAUCAFall dataset
    run_name = "caucafall_yolov8s"
    checkpoint = base_dir / "runs" / "detect" / run_name / "weights" / "last.pt"

    if checkpoint.exists():
        print(f"[INFO] Resuming 2-class training from: {checkpoint}")
        model = YOLO(str(checkpoint))
        resume_mode = True
    else:
        print("[INFO] Starting clean 2-class training with yolov8s.pt...")
        model = YOLO("yolov8s.pt")
        resume_mode = False

    # 4. Thermal & Kernel-Safe Training Configuration
    print("[TRAINING] Starting training on CAUCAFall with SGD optimizer...")
    model.train(
        data=str(yaml_path),
        epochs=50,             # 50 epochs gives solid convergence on CAUCAFall
        imgsz=640,
        batch=8,               # Optimal for 4GB RTX 3050 VRAM headroom
        device=device,
        workers=0,             # Windows-safe: avoids PyTorch multiprocessing shared-memory crash
        optimizer="SGD",       # Eliminates AdamW foreach memory spikes
        amp=False,             # Prevents FP16 driver drops on mobile RTX GPUs
        patience=12,           # Early stopping if mAP plateaus
        project=str(base_dir / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        resume=resume_mode,
        verbose=True
    )

    print("\n" + "="*60)
    print("TRAINING FINISHED!")
    print(f"Optimal weights saved at: {base_dir / 'runs' / 'detect' / run_name / 'weights' / 'best.pt'}")
    print("="*60)

if __name__ == "__main__":
    train_stable()