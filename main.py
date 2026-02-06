"""
Main script to test the CIFAR-10 generation competition locally.
"""

import numpy as np
import matplotlib.pyplot as plt

from env import Env
from agent_diffusion import Agent
from agent_diffusion_UNet import Agent
#from agent_example import Agent


def main():
    # Create environment and agent
    env = Env()
    agent = Agent()

    # Wrap agent in list (as expected by env.evaluate)
    agents = [agent]
    agent_infos = [{"agent_index": 0, "agent_name": "diffusion_agent"}]

    # Run evaluation
    print("Running evaluation...")
    results = env.evaluate(agents, agent_infos)
    fid_score = -results["agent_results"][0]["score"]
    print(f"\nFID: {fid_score:.2f}")

    # Generate and plot 3 sample images
    class_ids = np.array([0, 1, 2], dtype=np.int32)  # airplane, automobile, bird
    images = agent.generate(class_ids)

    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for idx, ax in enumerate(axes):
        # Convert from (3, 32, 32) to (32, 32, 3) for display
        img = np.transpose(images[idx], (1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"Class: {class_names[class_ids[idx]]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"sample_fid_{fid_score:.2f}.png")
    plt.show()


if __name__ == "__main__":
    main()
