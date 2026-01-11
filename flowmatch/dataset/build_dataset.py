from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def build_dataset(args):
    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        val_transform = transform
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=val_transform)
    elif args.dataset == "cifar":
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
        train_ds = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="../data", train=False, download=True, transform=val_transform)
    elif args.dataset == "imagenet100":
        # from https://github.com/cloneofsimo/imagenet.int8
        from datasets.data_utils import StreamingWrapperDataset
        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ["SLURM_JOB_ID"]
            user = os.environ.get("USER", "unknown_user")
            local_cache_path = f"/tmp/{user}/{job_id}/vae_mds_100"
            print(f"Environment: SLURM Detected. Caching to local SSD: {local_cache_path}")
        else:
            local_cache_path = "../local_cache_imagenet"
            print(f"Environment: Local Detected. Caching to: {local_cache_path}")

        train_ds = StreamingWrapperDataset(
            local=local_cache_path,
            remote=args.data_dir,
            split=None,
            shuffle=True,
            shuffle_algo="naive",
            num_canonical_nodes=1,
            batch_size=batch_size,
        )
        test_ds = train_ds
    return train_ds, test_ds


def build_dataloaders(args):
    init_fn = None
    shuffle = True
    if args.dataset == "imagenet100":
        from datasets.data_utils import worker_init_fn
        init_fn = worker_init_fn
        shuffle = False
    train_ds, test_ds = build_dataset(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=6, pin_memory=True, persistent_workers=True, worker_init_fn=init_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, worker_init_fn=init_fn)
    return train_loader, test_loader


def get_dataset_info(args):
    # H, W, C, num_classes
    if args.dataset == "mnist":
        return 32, 32, 1, 10
    elif args.dataset == "cifar":
        return 32, 32, 3, 10
    elif args.dataset == "imagenet100":
        return 32, 32, 4, 100