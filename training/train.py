import os
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast 
from data_pipeline.loader import EEGSpectrogramDataset
from models.model import build_model
from sklearn.metrics import accuracy_score

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on: {device}")
    
    train_ds = EEGSpectrogramDataset("dataset/processed/train", is_training=True)
    val_ds = EEGSpectrogramDataset("dataset/processed/val", is_training=False)
    
    # num_workers=0 to protect system RAM from float64 spikes
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True
    )
    
    model = build_model().to(device)
   # Reduced LR to 1e-5 to prevent the "Ping-Pong" effect
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # Stronger weight for the minority class (Control)
    class_weights = torch.tensor([2.5, 1.0], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    scaler = GradScaler('cuda')
    
    best_acc = 0.0
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                preds = model(x)
                loss = criterion(preds, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                with autocast('cuda'):
                    out = model(x_val)
                    loss = criterion(out, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(out.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
            print(f"   -> Model saved! New best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_pipeline()