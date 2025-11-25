Training guide for retraining the rice disease model

Files
- `train_transfer.py`: Transfer-learning script that trains a head on MobileNetV2, computes class-weights, and optionally fine-tunes the backbone.

Quick start (PowerShell)

1. Install dependencies (use your environment):
```powershell
pip install -r requirements.txt
# If you use tensorflow-addons or sklearn adjust accordingly
```

2. Run head training (defaults):
```powershell
python .\backend\train_transfer.py --train_dir data/train --val_dir data/val --epochs_head 12 --batch_size 32
```

3. Fine-tune (unfreeze last 30 layers) after head training:
```powershell
python .\backend\train_transfer.py --train_dir data/train --val_dir data/val --fine_tune --unfreeze_layers 30 --epochs_finetune 8
```

Outputs
- `model/retrained_mobilenetv2.h5` (best checkpoint)
- `backend/training_history.json` (training metrics)

Recommendations
- Clean dataset labels and remove corrupted files before training.
- Use class weights (script computes them automatically).
- Monitor per-class precision/recall and confusion matrix.
- Try stronger augmentations, MixUp/CutMix, or focal loss if classes remain confused.
