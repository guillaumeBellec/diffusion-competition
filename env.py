"""
CIFAR-10 Image Generation Competition Environment

Evaluates agents on their ability to generate class-conditional CIFAR-10 images.
Score is computed using FID (Fréchet Inception Distance) against the test set.
"""

import pickle
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


class Env:
    def __init__(self):
        """Load CIFAR-10 test set and initialize FID metric."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_images = 128
        self.batch_size = 32

        # Load CIFAR-10 test set from cached binary file
        with open('test_batch', 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')

        # Extract images: shape (10000, 3072) -> (10000, 3, 32, 32)
        images = test_data[b'data'].reshape(-1, 3, 32, 32)

        # Initialize FID and add real images in batches
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        for i in range(0, min(len(images), self.num_images), 16):
            batch = torch.from_numpy(images[i:i+16]).to(self.device)
            self.fid.update(batch, real=True)
            if self.fid.real_features_num_samples >= self.num_images:
                break

        # Prepare balanced class_ids for evaluation (1024 images, ~102 per class)
        self.class_ids_batches = self._prepare_class_ids()


    def _prepare_class_ids(self):
        """Prepare 16 batches of 64 balanced class IDs."""
        # 1024 total images, 102-103 per class
        all_class_ids = []

        for class_id in range(10):
            count = 103 if class_id < 4 else 102  # 4*103 + 6*102 = 1024
            all_class_ids.extend([class_id] * count)

        all_class_ids = np.array(all_class_ids, dtype=np.int32)
        np.random.shuffle(all_class_ids)

        # Split into 16 batches of 64
        num_batches = self.num_images // self.batch_size
        batch_size = self.batch_size
        return [all_class_ids[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

    def evaluate(self, agents: list, agent_infos: list) -> dict:
        """Evaluate agent on CIFAR-10 generation using FID."""
        results = []
        agent = agents[0]

        try:
            for class_ids in self.class_ids_batches:
                images = agent.generate(class_ids)
                self.fid.update(torch.from_numpy(images).to(self.device), real=False)
                if self.fid.fake_features_num_samples >= self.num_images: break

            fid_score = self.fid.compute().item()
            results.append({
                "agent_index": 0,
                "score": -fid_score,
                "info_message": f"FID: {fid_score:.2f}"
            })
        except Exception as e:
            results.append({
                "agent_index": 0,
                "score": -1e9,
                "is_agent_code_error": True,
                "agent_code_error_message": str(e)
            })

        return {"agent_results": results}
