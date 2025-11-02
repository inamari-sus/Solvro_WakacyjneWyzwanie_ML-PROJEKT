import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import itertools
from torch.optim import AdamW
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# DEFINICJA CALLBACKS
# ==============================================================================
class EarlyStopping:

    def __init__(self, patience=5, min_delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} z {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Strata walidacyjna zmalała ({self.best_loss:.6f} --> {val_loss:.6f}). Zapisywanie modelu...')
        torch.save(model.state_dict(), self.path)


# ==============================================================================
# DEFINICJA FUNKCJI
# ==============================================================================
def run_epoch(loader, model, criterion, device, train: bool, optimizer=None, scaler=None):
    if train and (optimizer is None or scaler is None):
        raise ValueError("Optimizer and Scaler must be provided for training.")

    epoch_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    model.train() if train else model.eval()

    desc = "Trening" if train else "Walidacja/Test"
    loader_with_tqdm = tqdm(loader, total=len(loader), desc=desc, leave=False, unit='batch')

    with torch.set_grad_enabled(train):
        for xb, yb in loader_with_tqdm:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=train):
                logits = model(xb)
                loss = criterion(logits, yb)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)


            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

            loader_with_tqdm.set_postfix(loss=loss.item(), acc=(preds == yb).float().mean().item())


    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    avg_loss = epoch_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, y_pred, y_true


def save_history_plots(history, plot_dir="plots"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Strata treningowa")
    plt.plot(history["val_loss"], label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.title("Wykres straty (Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_acc"], label="Dokładność treningowa")
    plt.plot(history["val_acc"], label="Dokładność walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność (Accuracy)")
    plt.title("Wykres dokładności (Accuracy)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "accuracy_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history["val_precision"], label="Precyzja")
    plt.plot(history["val_recall"], label="Czułość (Recall)")
    plt.plot(history["val_f1"], label="F1-Score")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość")
    plt.title("Metryki klasyfikacji na zbiorze walidacyjnym")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "classification_metrics_plot.png"))
    plt.close()


def save_confusion_matrix_plot(cm, class_names, plot_dir="plots", filename="confusion_matrix.png"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Macierz pomyłek")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Predykcja')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


def main():
    # ==============================================================================
    # KONFIGURACJA
    # ==============================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    data_dir = './data/patches1'
    EPOCHS = 5
    LR = 1e-4
    BATCH_SIZE = 128
    PATIENCE = 5
    PLOT_DIR = "plots"
    CHECKPOINT_PATH = 'checkpoint.pt'

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=CHECKPOINT_PATH)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    num_workers = min(os.cpu_count(), 8)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                 shuffle=(x == 'train'), num_workers=num_workers, pin_memory=True)
                   for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Liczba obrazów: {dataset_sizes}")
    print(f"Nazwy klas: {class_names}")

    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    scaler = GradScaler()

    optimizer = AdamW(model.parameters(), lr=LR, fused=True)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_precision": [], "val_recall": [], "val_f1": []}

    # ==============================================================================
    # PĘTLA TRENINGOWA
    # ==============================================================================
    start = time.time()
    print("\nRozpoczęto trenowanie...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoka {epoch:02d}/{EPOCHS}")
        print("-" * 15)

        tr_loss, tr_acc, _, _ = run_epoch(dataloaders['train'], model, criterion, device, train=True,
                                          optimizer=optimizer, scaler=scaler)

        va_loss, va_acc, y_pred_val, y_true_val = run_epoch(dataloaders['val'], model, criterion, device, train=False)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro',
                                                                   zero_division=0)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        print(
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"Val Precision {precision:.4f} Recall {recall:.4f} F1 {f1:.4f}"
        )

        early_stopping(va_loss, model)
        if early_stopping.early_stop:
            print("Wczesne zatrzymanie treningu.")
            break

    print(f"\nCzas treningu: {time.time() - start:.1f}s")

    # ==============================================================================
    # FINALNA EWALUACJA NA ZBIORZE TESTOWYM
    # ==============================================================================
    print("\nEwaluacja najlepszego modelu na zbiorze testowym...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print("Wczytano najlepszy model z punktu kontrolnego.")
    except FileNotFoundError:
        print("Nie znaleziono pliku checkpoint.pt. Ewaluacja na ostatnim modelu.")

    test_loss, test_acc, y_pred_test, y_true_test = run_epoch(dataloaders['test'], model, criterion, device,
                                                              train=False)
    print(f"\nWyniki na zbiorze testowym:")
    print(f"Strata (Loss): {test_loss:.4f}")
    print(f"Dokładność (Accuracy): {test_acc:.4f}")

    report = classification_report(y_true_test, y_pred_test, target_names=class_names, zero_division=0)
    print("\nRaport klasyfikacji:")
    print(report)

    # ==============================================================================
    # ZAPIS WYNIKÓW
    # ==============================================================================
    print("\nZapisywanie wykresów i macierzy pomyłek...")
    save_history_plots(history, PLOT_DIR)

    cm_test = confusion_matrix(y_true_test, y_pred_test)
    save_confusion_matrix_plot(cm_test, class_names, PLOT_DIR, filename="confusion_matrix_test.png")
    print("Zakończono. Wyniki zapisano w folderze 'plots'.")


# ==============================================================================
# PUNKT STARTOWY SKRYPTU
# ==============================================================================
if __name__ == '__main__':
    main()