import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions, labels_all = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_all, predictions),
    }


def train_model(model, loaders, criterion, optimizer, scheduler, device, max_epochs, patience):
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_metrics = run_epoch(model, loaders["validation"], criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"Epoch {epoch:02d}: train loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}; val loss={validation_metrics['loss']:.4f}, "
            f"acc={validation_metrics['accuracy']:.4f}"
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    model.load_state_dict(best_state)
    return history


def evaluate_model(model, loader, class_names, device):
    model.eval()
    labels_all, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            labels_all.extend(labels.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    report = classification_report(
        labels_all,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels_all, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            labels_all, predictions, labels=list(range(len(class_names)))
        ).tolist(),
    }


def benchmark_cpu(model, loader, warmup_batches=3, measured_batches=20):
    model.eval().cpu()
    samples = 0
    durations = []
    with torch.inference_mode():
        for index, (images, _) in enumerate(loader):
            start = time.perf_counter()
            model(images.cpu())
            elapsed = time.perf_counter() - start
            if index >= warmup_batches:
                durations.append(elapsed)
                samples += images.size(0)
            if len(durations) >= measured_batches:
                break
    total = sum(durations)
    return {
        "mean_batch_latency_ms": 1000 * float(np.mean(durations)) if durations else None,
        "throughput_images_per_second": samples / total if total else None,
        "measured_batches": len(durations),
    }

