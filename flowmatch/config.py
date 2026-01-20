import torch
from hydra.utils import instantiate
from torch.optim.lr_scheduler import LambdaLR
from embeddings.build_embeddings import build_embedding_provider
from dataset.build_dataset import build_dataloaders
from evaluation.train_classifier import Classifier
from utils import load_checkpoint
from models.utils import EMA

def _is_cross_modal(cfg):
    return cfg.mode.source == "label" or cfg.mode.target == "label"

def _get_model_internal_cls_dropout_prob(cfg):
    if cfg.use_conditioning and _is_cross_modal(cfg):
        return cfg.rf.cls_dropout_prob
    return 0.0

def _get_model_internal_num_classes(cfg):
    if cfg.use_conditioning and _is_cross_modal(cfg):
        return cfg.dataset.num_classes
    return 2

def build_rf(cfg):
    """
    Builds the RF class, should be used to instantiate the RF class
    """
    if cfg.mode.source == cfg.mode.target:
        raise ValueError("Source and target cannot be the same")
    H = cfg.dataset.image_size
    C = cfg.dataset.channels
    num_classes = cfg.dataset.num_classes

    num_classes_model = _get_model_internal_num_classes(cfg)
    cls_dropout_prob_model = _get_model_internal_cls_dropout_prob(cfg)

    model = instantiate(
        cfg.model,
        num_classes=num_classes_model,
        class_dropout_prob=cls_dropout_prob_model,
    )
    model.to(cfg.device)

    if cfg.device == "cuda" and cfg.compile_model:
        model = torch.compile(model, mode="max-autotune")

    label_embedder = None
    if _is_cross_modal(cfg):
        label_embedder = instantiate(cfg.label_embedding)
        label_embedder.to(cfg.device)

    rf = instantiate(
        cfg.rf,
        model=model,
        label_embedder=label_embedder,
        use_conditioning=False if _is_cross_modal(cfg) else cfg.use_conditioning
    )
    return rf


def setup(rf, cfg):
    ema = EMA(rf.model, decay=cfg.ema_decay, warmup_steps=cfg.ema_warmup_steps)
    scheduler = None
    optimizer = instantiate(cfg.optimizer, params=rf.model.parameters())

    if cfg.lr_warmup_steps > 0:
        warmup_steps = cfg.lr_warmup_steps
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)

    if cfg.resume_checkpoint:
        global_step = load_checkpoint(rf.model, optimizer, ema, scheduler, cfg.resume_checkpoint)
        global_step += 1
        print(f"Resuming from checkpoint: {cfg.resume_checkpoint} at step {global_step}")
    else:
        global_step = 0

    train_loader, test_loader = build_dataloaders(cfg)

    classifier = None
    if cfg.classifier_path:
        C = cfg.dataset.channels
        classifier = Classifier(in_channels=C).to(cfg.device)
        classifier.load_state_dict(torch.load(cfg.classifier_path, map_location=cfg.device))
        classifier.eval()

    return ema, optimizer, scheduler, global_step, train_loader, test_loader, classifier