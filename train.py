# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]    = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from ultralytics import YOLO

torch.cuda.empty_cache()

print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_YAML  = r"D:\data\data.yaml"          # ✅ dataset on D drive
MODEL      = "yolo11m.pt"
EPOCHS     = 150
IMG_SIZE   = 640
BATCH_SIZE = 8
PROJECT    = r"D:\yolo_runs\defect"        # ✅ all outputs saved to D drive
RUN_NAME   = "neu_yolo11m_best"

# ── Callbacks ──────────────────────────────────────────────────────────────────
def on_train_epoch_end(trainer):
    e        = trainer.epoch + 1
    epochs   = trainer.epochs
    box_loss = trainer.loss_items[0].item()
    cls_loss = trainer.loss_items[1].item()
    dfl_loss = trainer.loss_items[2].item()
    lr       = trainer.optimizer.param_groups[0]["lr"]
    mem      = torch.cuda.memory_reserved(0) / 1024**3

    print(f"\n📍 Epoch {e}/{epochs}")
    print(f"   📉 box_loss: {box_loss:.4f}  cls_loss: {cls_loss:.4f}  dfl_loss: {dfl_loss:.4f}")
    print(f"   🎯 lr: {lr:.6f}   🖥 VRAM: {mem:.2f} GB")

def on_val_end(validator):
    metrics   = validator.metrics
    map50     = metrics.results_dict.get("metrics/mAP50(B)",     0)
    map50_95  = metrics.results_dict.get("metrics/mAP50-95(B)",  0)
    precision = metrics.results_dict.get("metrics/precision(B)", 0)
    recall    = metrics.results_dict.get("metrics/recall(B)",    0)

    print(f"   ✅ mAP50: {map50:.3f}  mAP50-95: {map50_95:.3f}  P: {precision:.3f}  R: {recall:.3f}")
    print(f"   {'─'*50}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model = YOLO(MODEL)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end",         on_val_end)

    print(f"\n🔥 Training on  : {torch.cuda.get_device_name(0)}")
    print(f"📦 Model        : {MODEL}")
    print(f"🧊 Frozen layers: 0–9 (backbone)")
    print(f"⚡ Trainable    : 10–end (neck + head)\n")

    model.train(
        data            = DATA_YAML,
        epochs          = EPOCHS,
        imgsz           = IMG_SIZE,
        batch           = BATCH_SIZE,
        project         = PROJECT,
        name            = RUN_NAME,
        device          = 0,

        # ── Freezing ──────────────────────────────────────────────────
        freeze          = 10,

        # ── Learning rate ─────────────────────────────────────────────
        optimizer       = "AdamW",
        lr0             = 0.0005,
        lrf             = 0.01,
        cos_lr          = True,
        momentum        = 0.937,
        weight_decay    = 0.0005,

        # ── Warmup ────────────────────────────────────────────────────
        warmup_epochs   = 7.0,
        warmup_bias_lr  = 0.05,
        warmup_momentum = 0.8,

        # ── Regularization ────────────────────────────────────────────
        dropout         = 0.1,
        label_smoothing = 0.1,

        # ── Early stopping ────────────────────────────────────────────
        patience        = 40,

        # ── Augmentation ──────────────────────────────────────────────
        augment         = True,
        mosaic          = 1.0,
        close_mosaic    = 20,
        mixup           = 0.15,
        copy_paste      = 0.05,
        fliplr          = 0.5,
        flipud          = 0.1,
        degrees         = 10.0,
        translate       = 0.1,
        scale           = 0.5,
        shear           = 2.0,
        perspective     = 0.0001,
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,
        erasing         = 0.4,

        # ── Loss weights ──────────────────────────────────────────────
        box             = 7.5,
        cls             = 1.0,
        dfl             = 1.5,

        # ── System ────────────────────────────────────────────────────
        pretrained      = True,
        val             = True,
        save_period     = -1,           # ✅ only saves best.pt and last.pt
        plots           = False,
        cache           = False,        # ✅ no caching — saves RAM and disk
        workers         = 0,
        amp             = True,
    )

    print("\n✅ Training complete!")
    print(f"📁 Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")

    torch.cuda.empty_cache()

    print("\n📊 Running final validation with TTA...")
    metrics = model.val(
        data    = DATA_YAML,
        imgsz   = IMG_SIZE,
        batch   = 1,
        workers = 0,
        augment = True,                 # ✅ TTA — boosts mAP by 1-3%
    )
    print(f"\n📈 mAP50:     {metrics.box.map50:.3f}")
    print(f"📈 mAP50-95:  {metrics.box.map:.3f}")
    print(f"📈 Precision: {metrics.box.mp:.3f}")
    print(f"📈 Recall:    {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()

# %%



