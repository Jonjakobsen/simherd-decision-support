import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from src.model_gru import CowHealthGRU
from src.utils import PROCESSED_DATA_DIR, TRAINED_MODEL_DIR, ROOT_DIR
from sklearn.model_selection import train_test_split

def evaluate():
    # 1. Setup mapper til rapporter
    reports_dir = ROOT_DIR / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load data og genskab test-split
    data_path = PROCESSED_DATA_DIR / "processed_data.pth"
    if not data_path.exists():
        print("⚠️ Data mangler! Kør src/features.py først.")
        return
        
    data = torch.load(data_path, weights_only=False)
    X, y = data['X'], data['y']
    
    indices = np.arange(len(y))
    _, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])
    
    X_test, y_test = X[test_idx], y[test_idx]

    # 3. Load model
    model = CowHealthGRU()
    model_path = TRAINED_MODEL_DIR / "gru_model.pth"
    if not model_path.exists():
        print("⚠️ Model mangler! Kør src/model_gru.py først.")
        return
        
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 4. Inference
    with torch.no_grad():
        logits = model(X_test).squeeze()
        probs = torch.sigmoid(logits).numpy()
        y_pred = (probs > 0.5).astype(float)
        y_true = y_test.numpy()

    # 5. Visualisering
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Confusion Matrix Heatmap ---
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax1,
                xticklabels=['Rask', 'Syg'], yticklabels=['Rask', 'Syg'])
    ax1.set_title('Confusion Matrix (Test Sæt)')
    ax1.set_xlabel('Forudsagt Status')
    ax1.set_ylabel('Faktisk Status')

    # --- Plot 2: Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap_score = average_precision_score(y_true, probs)
    
    ax2.plot(recall, precision, color='forestgreen', lw=2, label=f'Average Precision = {ap_score:.2f}')
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall (Evne til at finde syge)')
    ax2.set_ylabel('Precision (Sikkerhed ved alarm)')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Gem figuren
    plot_path = reports_dir / "model_evaluation.png"
    plt.savefig(plot_path)
    print(f"✅ Plots er gemt i: {plot_path}")
    plt.close()

    # 6. Print tekst-rapport
    print("\n" + "="*40)
    print("📋 DETALJERET KLASSIFIKATIONSRAPPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=["Rask", "Syg"]))

if __name__ == "__main__":
    evaluate()