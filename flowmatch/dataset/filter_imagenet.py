import argparse
import shutil
import os
import json
import numpy as np
from tqdm import tqdm
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
    np.random.seed(0)
    all_classes = np.arange(total_classes)
    
    if num_to_keep >= total_classes:
        return set(all_classes)
        
    return set(np.random.choice(all_classes, num_to_keep, replace=False))

def create_filtered_dataset(src_local, dst_local, keep_classes, num_workers=2, batch_size=512):
    if os.path.exists(dst_local):
        print(f"Cleaning up existing destination: {dst_local}")
        shutil.rmtree(dst_local)

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

    print(f"Processing dataset...")
    print(f"Source: {src_local}")
    print(f"Filter: Keeping {len(target_indices)} classes")
    print(f"Workers: {num_workers} | Batch Size: {batch_size}")

    with MDSWriter(out=dst_local, columns=columns) as out:
        for batch in tqdm(loader, total=len(dataset) // batch_size):
            for sample in batch:
                original_label = int(sample['label'])
                
                if original_label in label_map:
                    new_sample = sample.copy()
                    new_sample['label'] = label_map[original_label]
                    out.write(new_sample)

    mapping_path = os.path.join(dst_local, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        save_map = {v: int(k) for k, v in label_map.items()}
        json.dump(save_map, f, indent=4)

    print(f"Success! Class mapping saved to {mapping_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter MosaicML Streaming Dataset.")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for reading")
    
    args = parser.parse_args()
    
    create_filtered_dataset(args.src, args.dst, args.classes, num_workers=args.workers)