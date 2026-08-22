"""Fine-tune a pretrained charter checkpoint with pitch conditioning.

    python tools/finetune.py --data data --out runs/pitch-m
    python tools/finetune.py --data data --model 3podi/charter-v1.0-40-S-best-acc --out runs/pitch-s

Starts from the author's pretrained weights (the same HF checkpoint inference
uses — export_checkpoint.py proves the training and inference state dicts are
interchangeable) and adds the zero-initialised PitchConditioner, so step 0 is
exactly the pretrained model. What trains is decided by --unfreeze:

    conditioner  only the ~52k-param pitch projection. Cheapest, but a frozen
                 decoder was trained to read Encodec features, not this new
                 channel — expect little movement (kept for the ablation).
    xattn        conditioner + every cross-attention block + the audio memory
                 norm + the output projection (default). The decoder re-learns
                 how to *read* its audio memory, which now carries pitch,
                 without disturbing its rhythm/pattern knowledge.
    all          full fine-tune. With a ~116-song corpus this will overfit;
                 use only with a much larger dataset.

Outputs, under --out:
    checkpoints/…                 Lightning checkpoints (best on val acc)
    export/config.json            HF-style folder Charter/PitchCharter can load
    export/pytorch_model.bin
    export/conditioner.pt         PitchConditioner weights for inference

VRAM: M at batch 4, bf16, frozen encoder measures well inside 8 GB because
only cross-attn activations need gradients. Drop --batch to 2 if it OOMs.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "audio2chart"))

UNFREEZE_CHOICES = ("conditioner", "xattn", "all")
MONITORED_METRIC = "val/acc_nonpad_epoch"


def build_config(args, hf_config: dict):
    """Mirror the HF checkpoint's architecture into a hydra-style config."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "seed": args.seed,
        "model": {
            "name": f"pitchft_{Path(args.model).name}",
            "freeze_encoder": True,
            "use_processor": True,
            "use_pitch": True,
            "sample_rate": 24000,
            "encoder": {
                "_target_": "modules.models.Encodec",
                "checkpoint": "facebook/encodec_24khz",
                "bandwidth": 3.0,
                "sample_rate": 24000,
            },
            "transformer": {
                "_target_": "modules.training_transformer.TransformerDecoderAudioConditioned",
                **{k: hf_config[k] for k in (
                    "d_model", "d_ff", "n_heads", "num_kv_heads", "n_layers",
                    "dropout", "audio_drop", "compression", "rope_base",
                    "conditional", "use_flash",
                )},
            },
        },
        "data": {
            "root_folder": str(args.data),
            "split_folder": None,
            "validation_split": args.val_split,
            "difficulties": ["Expert"],
            "instruments": ["Single"],
            "max_length": 2048,
            "error_policy": "skip",
            "window_seconds": 30,
            "grid_ms": hf_config["grid_ms"],
        },
        "loader": {
            "batch_size": args.batch,
            "val_batch_size": args.batch,
            "num_workers": args.workers,
            "train_num_pieces": args.pieces,
            "val_num_pieces": 1,
            "max_sample_retries": 5,
            "augment": False,
        },
        "optimizer": {
            "lr": args.lr,
            # Read unconditionally by configure_optimizers even though the
            # encoder is frozen and never gets a param group.
            "lr_audio": args.lr,
            "weight_decay": 0.01,
            "warmup_steps": args.warmup,
            "max_steps": args.steps,
        },
        "trainer": {
            "max_epochs": args.epochs,
            "gpus": 1 if args.gpu else 0,
            "precision": "bf16-mixed" if args.gpu else "32",
            "save_run": True,
            "log_every_n_steps": 10,
            "early_stopping_patience": args.patience,
            "gradient_clip_val": 1.0,
            "num_sanity_val_steps": 1,
        },
        "tracking": {"enabled": False, "project": "chartgen", "tags": ["pitch"]},
        "logging": {"level": "INFO"},
    })


def apply_freeze_policy(model, policy: str) -> tuple[int, int]:
    """Freeze the transformer except what the policy names. Returns
    (trainable, total) parameter counts for the report."""
    keep = {
        "conditioner": (),
        "xattn": ("cross_attn", "norm_audio", "output_projection"),
        "all": None,  # everything trains
    }[policy]

    if keep is not None:
        for name, param in model.transformer.named_parameters():
            param.requires_grad = any(k in name for k in keep)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_pretrained(model, repo_or_dir: str) -> dict:
    """Load HF-format charter weights into the training transformer.

    Returns the checkpoint's config dict. Accepts a hub repo id or a local
    directory containing config.json + pytorch_model.bin.
    """
    import torch

    local = Path(repo_or_dir)
    if local.is_dir():
        cfg_path, bin_path = local / "config.json", local / "pytorch_model.bin"
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_or_dir, "config.json")
        bin_path = hf_hub_download(repo_or_dir, "pytorch_model.bin")

    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    state = torch.load(bin_path, map_location="cpu")
    # strict=True is the point: a silent partial load would "train" from
    # damaged weights and the run would look fine until playtesting.
    model.transformer.load_state_dict(state, strict=True)
    return config


def export_for_inference(model, hf_config: dict, out_dir: Path) -> None:
    import torch

    export = out_dir / "export"
    export.mkdir(parents=True, exist_ok=True)
    torch.save(model.transformer.state_dict(), export / "pytorch_model.bin")
    (export / "config.json").write_text(
        json.dumps(hf_config, indent=2) + "\n", encoding="utf-8"
    )
    if model.pitch_conditioner is not None:
        torch.save(model.pitch_conditioner.state_dict(), export / "conditioner.pt")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=Path("data"),
                    help="folder from tools/build_dataset.py (with pitch/)")
    ap.add_argument("--model", default="3podi/charter-v1.0-40-M-best-acc")
    ap.add_argument("--out", type=Path, default=Path("runs") / "pitch")
    ap.add_argument("--unfreeze", choices=UNFREEZE_CHOICES, default="xattn")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="fine-tune LR; the pretrained weights are good, do "
                         "not blast them with the from-scratch 1e-3")
    ap.add_argument("--steps", type=int, default=6000, help="LR schedule horizon")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6,
                    help="early-stop epochs without val improvement")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pieces", type=int, default=1,
                    help="windows sampled per song per epoch pass. NOTE: the "
                         "collator flattens, so VRAM sees batch*pieces windows "
                         "per step — 4x4 OOMed the 8 GB card; keep the product "
                         "at or under 4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", dest="gpu", action="store_false",
                    help="CPU training (only sensible for the S model)")
    ap.add_argument("--smoke", action="store_true",
                    help="two tiny batches through the full loop, then exit; "
                         "catches wiring mistakes before burning GPU hours")
    args = ap.parse_args()

    manifest = args.data / "audio_dataset_with_raw.json"
    if not manifest.is_file():
        ap.error(f"no manifest at {manifest} — run tools/build_dataset.py first")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    missing_pitch = [e for e in entries if "pitch_path" not in e]
    if missing_pitch:
        ap.error(f"{len(missing_pitch)} manifest entries lack pitch_path — "
                 f"rebuild the dataset without --no-pitch")

    import torch
    from main import build_dataloaders, load_data_splits
    from modules.run_utils import build_trainer, configure_logging, experiment_logger
    from modules.trainer import WaveformTransformerDiscrete
    from modules.utils_train import set_seed_everything

    configure_logging("INFO")
    set_seed_everything(args.seed)

    # Peek at the checkpoint config first: the model must be built to match it.
    probe = WaveformTransformerDiscrete  # noqa: F841  (import order clarity)
    if Path(args.model).is_dir():
        hf_config = json.loads((Path(args.model) / "config.json").read_text())
    else:
        from huggingface_hub import hf_hub_download
        hf_config = json.loads(Path(hf_hub_download(args.model, "config.json")).read_text())
    if hf_config["grid_ms"] != int(entries[0].get("grid_ms", hf_config["grid_ms"])):
        ap.error(f"dataset grid_ms {entries[0].get('grid_ms')} does not match "
                 f"checkpoint grid_ms {hf_config['grid_ms']}")

    config = build_config(args, hf_config)
    if args.smoke:
        config.loader.train_num_pieces = 1
        config.loader.val_num_pieces = 1
        config.trainer.max_epochs = 1

    train_files, val_files = load_data_splits(config)
    train_loader, val_loader, vocab = build_dataloaders(config, train_files, val_files)

    # The checkpoint's token ids are the contract; the tokenizer must agree.
    for key, value in (("<bos>", hf_config["bos_token_id"]),
                       ("<eos>", hf_config["eos_token_id"]),
                       ("<PAD>", hf_config["pad_token_id"])):
        if vocab[key] != value:
            raise ValueError(f"tokenizer {key}={vocab[key]} but checkpoint "
                             f"expects {value}")

    model = WaveformTransformerDiscrete(
        pad_token_id=vocab["<PAD>"],
        eos_token_id=vocab["<eos>"],
        vocab_size=len(vocab),
        cfg_model=config.model,
        cfg_optimizer=config.optimizer,
    )
    load_pretrained(model, args.model)
    trainable, total = apply_freeze_policy(model, args.unfreeze)
    print(f"policy={args.unfreeze}: {trainable:,} of {total:,} params train "
          f"({trainable / total * 100:.1f}%)")

    if args.smoke:
        batch = next(iter(train_loader))
        with torch.no_grad():
            loss = model._step(batch, 0, "val")
        print(f"smoke: forward ok, loss {float(loss):.4f}, "
              f"pitch in batch: {'pitch' in batch}, "
              f"pitch shape: {tuple(batch['pitch'].shape) if 'pitch' in batch else '-'}")
        loss = model._step(batch, 0, "train")
        loss.backward()
        grads = sum(1 for p in model.parameters()
                    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        print(f"smoke: backward ok, {grads} parameter tensors received gradient")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    with experiment_logger(config, f"pitchft-{args.unfreeze}") as logger:
        trainer = build_trainer(config, logger, MONITORED_METRIC)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Export the best checkpoint (not the last step) for inference.
        best = None
        for callback in trainer.callbacks:
            if hasattr(callback, "best_model_path") and callback.best_model_path:
                best = callback.best_model_path
        if best:
            print(f"loading best checkpoint {best}")
            state = torch.load(best, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            shutil.copy2(best, args.out / "best.ckpt")
    export_for_inference(model, hf_config, args.out)
    print(f"exported to {args.out / 'export'} — use with:\n"
          f"  python -m chartgen song.mp3 --model \"{args.out / 'export'}\" "
          f"--conditioner \"{args.out / 'export' / 'conditioner.pt'}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
