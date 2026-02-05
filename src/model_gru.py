import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from src.utils import PROCESSED_DATA_DIR, TRAINED_MODEL_DIR

class CowHealthGRU(nn.Module):
    """GRU-arkitektur til detektion af sygdom i laktationsdata."""
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2):
        super(CowHealthGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialiser hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass gennem GRU
        out, _ = self.gru(x, h0)
        
        # Brug kun output fra sidste time step
        out = self.fc(out[:, -1, :])

        return out

def train_model(epochs=100, lr=0.001, batch_size=64, patience=5):
    # 1. Load data
    processed_path = PROCESSED_DATA_DIR / "processed_data.pth"
    if not processed_path.exists():
        print("⚠️ Processed data ikke fundet. Kør 'python -m src.features' først.")
        return

    data = torch.load(processed_path, weights_only=False)
    X, y = data['X'], data['y']
    full_dataset = TensorDataset(X, y)
    
    # 2. Stratificeret Split (80% træning, 10% validering, 10% test)
    # Jeg bruger stratify=y for at sikre ensartet sygdomsprævalens i alle splits
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx]
    )
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size)
    
    # 3. Model & Optimering
    model = CowHealthGRU()
    
    # Beregn vægt til Loss pga. ubalance (Positiv vægt = antal negative / antal positive)
    y_train = y[train_idx]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # BRUG pos_weight i loss function
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Early Stopping variabler
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"🚀 Starter træning (Samples: {len(train_idx)}, Patience: {patience})...")

    for epoch in range(epochs):
        # --- TRÆNING ---
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDERING ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                v_loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += v_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Early Stopping tjek
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Gem den foreløbigt bedste model
            torch.save(best_model_wts, TRAINED_MODEL_DIR / "gru_model.pth")
            msg = "⭐ Ny bedste model!"
        else:
            epochs_no_improve += 1
            msg = f"Ingen forbedring ({epochs_no_improve}/{patience})"

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | {msg}")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping ved epoch {epoch+1}. Bedste Val Loss: {best_val_loss:.5f}")
            break

    print(f"✅ Træning fuldført. Bedste model gemt i {TRAINED_MODEL_DIR}")

if __name__ == "__main__":
    train_model()