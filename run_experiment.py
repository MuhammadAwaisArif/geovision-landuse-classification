import argparse
import copy
from pathlib import Path

import torch
from torch import nn, optim

from geovision.compression import dynamically_quantize, model_statistics, prune_unstructured
from geovision.data import create_dataloaders
from geovision.engine import benchmark_cpu, evaluate_model, train_model
from geovision.models import create_model
from geovision.reporting import save_confusion_matrix
from geovision.utils import read_config, seed_everything, write_json


def optimizer_for(model_name, model, config):
    if model_name == "cnn":
        return optim.Adam(model.parameters(), lr=config["cnn_learning_rate"])
    learning_rate = (
        config["vit_learning_rate"] if model_name == "vit" else config["hybrid_learning_rate"]
    )
    return optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"]
    )


def evaluate_variant(name, model, test_loader, classes, output_dir):
    device = torch.device("cpu")
    model = model.cpu()
    metrics = evaluate_model(model, test_loader, classes, device)
    checkpoint = output_dir / f"{name}.pth"
    metrics["model"] = model_statistics(model, checkpoint)
    metrics["cpu_benchmark"] = benchmark_cpu(model, test_loader)
    write_json(metrics, output_dir / f"{name}_metrics.json")
    save_confusion_matrix(
        metrics["confusion_matrix"], classes, output_dir / f"{name}_confusion_matrix.png"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", choices=("cnn", "vit", "hybrid"), default="hybrid")
    parser.add_argument("--compress", action="store_true", help="Evaluate hybrid compression variants")
    args = parser.parse_args()

    if args.compress and args.model != "hybrid":
        parser.error("Compression study is defined only for the hybrid model")

    config = read_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = create_dataloaders(config, args.model, device)
    output_dir = Path(config["output_dir"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(args.model, len(classes)).to(device)
    optimizer = optimizer_for(args.model, model, config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"]
    )
    history = train_model(
        model,
        loaders,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        device,
        config["max_epochs"],
        config["early_stopping_patience"],
    )
    write_json({"history": history, "classes": classes}, output_dir / "training_history.json")

    best_path = output_dir / "best_model.pth"
    torch.save(model.state_dict(), best_path)
    baseline = copy.deepcopy(model).cpu()
    evaluate_variant("baseline", baseline, loaders["test"], classes, output_dir)

    if args.compress:
        # Every variant starts from the same freshly restored best hybrid weights.
        fresh = create_model("hybrid", len(classes), pretrained=False)
        fresh.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        pruned = prune_unstructured(fresh, 0.30)
        evaluate_variant("pruned_30pct", pruned, loaders["test"], classes, output_dir)
        evaluate_variant(
            "quantized_dynamic_linear", dynamically_quantize(fresh), loaders["test"], classes, output_dir
        )
        evaluate_variant(
            "pruned_30pct_quantized_dynamic_linear",
            dynamically_quantize(pruned),
            loaders["test"],
            classes,
            output_dir,
        )


if __name__ == "__main__":
    main()

