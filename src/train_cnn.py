import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# =========================
# CONFIGURATION
# =========================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================
# MODEL DEFINITION
# =========================
class ImprovedPlaque1DCNN(nn.Module):
    def __init__(self, num_tab=15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + num_tab, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, curves, tabular):
        cnn_out = self.cnn(curves).squeeze(-1)
        combined = torch.cat([cnn_out, tabular], dim=1)
        return self.fc(combined)

class PlaqueDataset(Dataset):
    def __init__(self, curves, tabular, labels):
        self.curves = torch.tensor(curves, dtype=torch.float32)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.curves[idx], self.tabular[idx], self.labels[idx]

def train_cnn_model(data_dir="data", model_dir="models", epochs=50):
    print("🧠 Training 1D-CNN + Tabular Fusion Model...")
    os.makedirs(model_dir, exist_ok=True)

    metadata_df = pd.read_csv(os.path.join(data_dir, "synthetic_lipospec_dataset_metadata.csv"))
    hdl_curves = np.load(os.path.join(data_dir, "hdl_curves.npy"))
    ldl_curves = np.load(os.path.join(data_dir, "ldl_curves.npy"))

    X_tab = metadata_df[["age","sex_male","bmi","diabetes","hypertension","smoking",
                         "hdl2b_percent","hdl2a_percent","hdl3a_percent","hdl3b_percent",
                         "hdl3c_percent","sdldl_percent","total_hdl","total_ldl","hdl_ldl_ratio"]].values
    X_curves = np.stack([hdl_curves, ldl_curves], axis=1)
    y = metadata_df["has_plaque"].values

    X_tab_train, X_tab_test, X_cur_train, X_cur_test, y_train, y_test = train_test_split(
        X_tab, X_curves, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(PlaqueDataset(X_cur_train, X_tab_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(PlaqueDataset(X_cur_test, X_tab_test, y_test), batch_size=32)

    model = ImprovedPlaque1DCNN(num_tab=15).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for curves, tab, labels in train_loader:
            curves, tab, labels = curves.to(DEVICE), tab.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(curves, tab)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for curves, tab, labels in test_loader:
                curves, tab, labels = curves.to(DEVICE), tab.to(DEVICE), labels.to(DEVICE)
                outputs = model(curves, tab)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_plaque_1dcnn_improved.pth"))

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Val AUC: {val_auc:.4f}")

    print(f"✅ Training complete. Best Val AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_cnn_model()
