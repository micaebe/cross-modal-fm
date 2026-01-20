from embeddings.label_embeddings import GrayScaleEmbedding, RectangleEmbedding, ClipEmbedding, SmoothRandom
import inspect

EMBEDDINGS = {
    "grayscale": GrayScaleEmbedding,
    "rectangle": RectangleEmbedding,
    "smoothrand": SmoothRandom,
    "clip": ClipEmbedding,
}


def build_embedding_provider(name: str, **kwargs):
    if name not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding provider: {name}")
    emb = EMBEDDINGS[name]
    sig = inspect.signature(emb)
    filtered_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }
    return emb(**filtered_args)