import json
import os

filepath = "tmp_batch_stats/rush_tune/confirm_tsm_only_056_040_raw_30_40.json"
output_metrics = "tmp_batch_stats/rush_tune/metrics_confirm_tsm_only_056_040_raw_30_40.json"
output_png = "tmp_batch_stats/rush_tune/confusion_matrix_confirm_tsm_only_056_040_raw_30_40.png"

with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

v_sum = data["violence_summary"]
nv_sum = data["nonviolence_summary"]

tp = v_sum["positive_videos"]
fn = v_sum["total_ok"] - tp
fp = nv_sum["positive_videos"]
tn = nv_sum["total_ok"] - fp

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

metrics = {
    "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "specificity": specificity
}

with open(output_metrics, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Non-Violence", "Violence"]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title="Confusion Matrix",
           ylabel="True Label",
           xlabel="Predicted Label")
           
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.savefig(output_png)
    print(f"Confusion matrix saved to {output_png}")
except Exception as e:
    print(f"Matplotlib failed: {e}")
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((50, 50), "Confusion Matrix", fill=(0,0,0))
        d.text((50, 100), f"TN: {tn}  FP: {fp}", fill=(0,0,0))
        d.text((50, 150), f"FN: {fn}  TP: {tp}", fill=(0,0,0))
        d.text((50, 250), f"Acc: {accuracy:.4f}", fill=(0,0,0))
        d.text((50, 280), f"F1 : {f1:.4f}", fill=(0,0,0))
        img.save(output_png)
        print(f"Simple image saved to {output_png}")
    except Exception as e2:
        print(f"PIL also failed: {e2}")
