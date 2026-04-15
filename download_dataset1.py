from datasets import load_from_disk
from torchvision import transforms

dataset = load_from_disk("/public/home/ssjxkzk/imagenette_dataset")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

train_dataset.set_transform(lambda x: {
    "pixel_values": transform(x["image"]),
    "label": x["label"]
})