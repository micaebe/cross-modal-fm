from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

def build_dataset(cfg):
    if cfg.dataset.name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        val_transform = transform
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=val_transform)
    elif cfg.dataset.name == "cifar":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
    elif cfg.dataset.name == "imagenet100":
        # from https://github.com/cloneofsimo/imagenet.int8
        from dataset.data_utils import StreamingWrapperDataset
        from streaming.base.util import clean_stale_shared_memory

        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ["SLURM_JOB_ID"]
            user = os.environ.get("USER", "unknown_user")
            path = os.environ.get("SLURM_TMPDIR", f"/tmp/{user}/{job_id}")
            local_cache_path = os.path.join(path, "vae_mds_100")
        else:
            local_cache_path = "./data/local_cache_imagenet"
        print(f"Caching to: {local_cache_path}")

        ds_kwargs = dict(
            #local=local_cache_path,
            remote=cfg.dataset.data_dir,
            split=None,
            shuffle=True,
            shuffle_algo="py1s",
            shuffle_seed=cfg.seed,
            num_canonical_nodes=1,
            batch_size=cfg.batch_size,
            shuffle_block_size=3000
        )
        # very hacky and inefficient workaround to avoid shared memory issues
        clean_stale_shared_memory()
        # for FID calculation we use a fixed "eval" set from the training data
        temp_ds = StreamingWrapperDataset(local=local_cache_path + "/temp", **ds_kwargs)
        test_samples = []
        test_labels = []
        iter_ds = iter(temp_ds)
        
        for _ in range(cfg.eval_batches * cfg.eval_batch_size):
            try:
                data = next(iter_ds)
                test_samples.append(data[0])
                test_labels.append(data[1])
            except StopIteration:
                break
        
        test_x = torch.stack(test_samples)
        test_y = torch.tensor(test_labels)
        test_ds = TensorDataset(test_x, test_y)
        del iter_ds
        del temp_ds
        clean_stale_shared_memory()
        train_ds = StreamingWrapperDataset(local=local_cache_path + "/train", **ds_kwargs)
    return train_ds, test_ds


def build_dataloaders(cfg):
    init_fn = None
    shuffle = True
    if cfg.dataset.name == "imagenet100":
        # in case of imagenet we dont shuffle in the dataloader but in the StreamingDataset
        from dataset.data_utils import worker_init_fn
        # worker init function is needed to convert from int8 to fp32
        init_fn = worker_init_fn
        shuffle = False
    train_ds, test_ds = build_dataset(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=init_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, num_workers=1, worker_init_fn=init_fn, persistent_workers=True)
    return train_loader, test_loader
