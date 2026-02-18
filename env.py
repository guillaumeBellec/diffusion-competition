"""
CIFAR-10 Image Generation Competition Environment

Evaluates agents on their ability to generate class-conditional CIFAR-10 images.
Score is computed using ResNetFID (Fréchet distance in ResNet18 feature space) against the test set.
"""

import pickle
import numpy as np
import torch
from eval import ResNetFID


class Env:
    def __init__(self):
        """Load CIFAR-10 test set and initialize ResNetFID metric."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_images = 1024
        self.batch_size = 64

        # Load CIFAR-10 test set
        with open('test_batch', 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
        self.real_images = torch.from_numpy(
            test_data[b'data'].reshape(-1, 3, 32, 32)[:self.num_images]
        )

        self.fid_metric = ResNetFID(device=self.device)
        self.class_ids_batches = self._prepare_class_ids()

    def _prepare_class_ids(self):
        """Prepare balanced class ID batches."""
        all_class_ids = []
        for class_id in range(10):
            count = self.num_images // 10
            all_class_ids.extend([class_id] * count)
        # Fill remainder
        for class_id in range(self.num_images - len(all_class_ids)):
            all_class_ids.append(class_id)

        all_class_ids = np.array(all_class_ids, dtype=np.int32)
        np.random.shuffle(all_class_ids)

        num_batches = self.num_images // self.batch_size
        return [all_class_ids[i*self.batch_size:(i+1)*self.batch_size] for i in range(num_batches)]

    def evaluate(self, agents: list, agent_infos: list) -> dict:
        """Evaluate agent on CIFAR-10 generation using ResNetFID."""
        results = []
        agent = agents[0]

        try:
            fake_images = []
            for class_ids in self.class_ids_batches:
                images = agent.generate(class_ids)
                fake_images.append(torch.from_numpy(images))
            fake_images = torch.cat(fake_images)[:self.num_images]

            fid_score = self.fid_metric.compute_fid(self.real_images, fake_images)
            results.append({
                "agent_index": 0,
                "score": -fid_score,
                "info_message": f"ResNetFID: {fid_score:.2f}"
            })
        except Exception as e:
            results.append({
                "agent_index": 0,
                "score": -1e9,
                "is_agent_code_error": True,
                "agent_code_error_message": str(e)
            })

        return {"agent_results": results}
