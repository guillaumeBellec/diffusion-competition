"""
Main script to test the CIFAR-10 generation competition locally.
Usage: python main.py <model_file.pth>
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from env import Env


def load_agent(model_path):
    """Load the correct agent class based on the model filename."""
    if model_path.startswith("flow_"):
        from agent_flow import Agent
    elif model_path.startswith("diffusion_UNet_"):
        from agent_diffusion_UNet import Agent
    else:
        from agent_diffusion import Agent
    return Agent(model_path)


def main():
    parser = argparse.ArgumentParser(description="Test CIFAR-10 generation agent")
    parser.add_argument("model", help="Path to model checkpoint (.pth)")
    args = parser.parse_args()

    env = Env()
    agent = load_agent(args.model)

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
