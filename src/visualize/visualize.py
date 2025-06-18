import random

from torch.utils.data import DataLoader

from datasets.mnist_dataset import MnistDataset
from utils.visualize_helper import plot_multiple_gray_images, plot_gray_image

def visualize_images():

    dataset = MnistDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False,
                            collate_fn=dataset.collate_fn)

    first_batch = next(iter(dataloader))
    images = first_batch["images"]
    labels = first_batch["targets"]
    ids = first_batch["ids"]

    titles = [f"id_{id}/label_{label}" for id, label in zip(ids, labels)]

    plot_multiple_gray_images(images, titles=titles, display=True)

def visualize_image(id: int):
    dataset = MnistDataset()
    
    sample = dataset[id]
    image = sample["image"]
    label = sample["target"]
    id = sample["id"]

    title = f"id_{id}/label_{label}"
    plot_gray_image(image=image, title=title)

if __name__ == "__main__":
    # visualize_images()
    visualize_image(0)
