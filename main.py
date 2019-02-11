import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms

from lib import models
from lib.core import inference

# from lib.dataset import coco


def main():
    model = models.get_fully_pretrained_pose_net()
    model.eval()
    print(model)

    image_file = Path("data/coco/val2017/000000581781.jpg")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    # cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    data_numpy = cv2.imread(
        str(image_file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )

    input = transform(data_numpy)
    output = model(input[None, :, :, :]).detach().numpy()
    print(inference.get_max_preds(output))


if __name__ == "__main__":
    main()
