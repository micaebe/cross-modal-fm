import argparse
import shutil
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from streaming import StreamingDataset, MDSWriter
from streaming.base.format.mds.encodings import Encoding, _encodings

class RawUInt8(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint8)

_encodings["uint8"] = RawUInt8

def get_target_indices(num_to_keep, total_classes=1000):
    if num_to_keep >= total_classes:
        return set(np.arange(total_classes))
    indices = np.linspace(0, total_classes - 1, num_to_keep)
    return set(np.round(indices).astype(int))

def create_filtered_dataset(src_local, dst_train, dst_test, keep_classes, 
                            test_samples_per_class=100, num_workers=2, batch_size=512):
    for dst in [dst_train, dst_test]:
        if os.path.exists(dst):
            print("Train and/or test directories already exists. Skipping")
            return

    dataset = StreamingDataset(local=src_local, batch_size=batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )
    
    columns = {'vae_output': 'uint8', 'label': 'int'}

    target_indices = get_target_indices(keep_classes)
    sorted_indices = sorted(list(target_indices))
    label_map = {old_id: new_id for new_id, old_id in enumerate(sorted_indices)}
    test_counts = defaultdict(int)

    print(f"Processing dataset...")
    print(f"Source: {src_local}")
    print(f"Train destination: {dst_train}")
    print(f"Test destination: {dst_test}")
    print(f"Filter: Keeping {len(target_indices)} classes")
    print(f"Test samples per class: {test_samples_per_class}")
    print(f"Workers: {num_workers} | Batch Size: {batch_size}")

    train_count = 0
    test_count = 0

    with MDSWriter(out=dst_train, columns=columns) as train_out, \
        MDSWriter(out=dst_test, columns=columns) as test_out:

        for batch in tqdm(loader, total=len(dataset) // batch_size):
            for sample in batch:
                original_label = int(sample['label'])
                
                if original_label in label_map:
                    new_sample = sample.copy()
                    new_sample['label'] = label_map[original_label]
                    if test_counts[original_label] < test_samples_per_class:
                        test_out.write(new_sample)
                        test_counts[original_label] += 1
                        test_count += 1
                    else:
                        train_out.write(new_sample)
                        train_count += 1
    
    for dst in [dst_train, dst_test]:
        mapping_path = os.path.join(dst, 'class_mapping.json')
        with open(mapping_path, 'w') as f:
            save_map = {v: int(k) for k, v in label_map.items()}
            json.dump(save_map, f, indent=4)
    
    print(f"Train samples: {train_count}")
    print(f"Test samples: {test_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter MosaicML Streaming Dataset with train/test split.")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst_train", type=str, required=True, help="Destination for train split")
    parser.add_argument("--dst_test", type=str, required=True, help="Destination for test split")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=100, help="Number of test samples per class")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for reading")

    args = parser.parse_args()

    create_filtered_dataset(args.src, args.dst_train, args.dst_test, args.classes, test_samples_per_class=args.test_samples, num_workers=args.workers)