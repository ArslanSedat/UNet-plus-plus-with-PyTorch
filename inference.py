import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from unetplusplus import UNetPlusPlus

def single_image_inference(image_path, model_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNetPlusPlus(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(tensor).argmax(dim=1)[0].cpu()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="viridis", vmin=0, vmax=2)
    for axis in axes:
        axis.axis("off")
    plt.show()

if __name__ == "__main__":
    raise SystemExit(
        "Call single_image_inference(image_path, model_path) after setting "
        "your image and checkpoint paths.")