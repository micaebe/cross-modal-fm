import torchvision.transforms as T
from torchvision import datasets

def get_transforms(dataset: str = "mnist") -> tuple[T.Compose, T.Compose]:
    if dataset == "mnist":
        train_trafo = T.Compose([
            #T.RandomRotation(degrees=10, fill=0),
            #T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        test_trafo = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        return train_trafo, test_trafo
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    

def get_datasets(dataset: str = "mnist"):
    train_trafo, test_trafo = get_transforms(dataset)
    if dataset == "mnist":
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_trafo)
        test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=test_trafo)
        return train_ds, test_ds, (1, 28, 28), 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")