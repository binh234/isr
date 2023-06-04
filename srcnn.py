from typing import Union
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms as T


class SRCNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        input_size=33,
        label_size=21,
        scale=2,
        device=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.label_size = label_size
        self.pad = (self.input_size - self.label_size) // 2
        self.scale = scale
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 9),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, 5),
            nn.ReLU(),
        )
        self.transform = T.Compose(
            [T.ToTensor()]  # Scale between [0, 1]
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def pre_process(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x / 255.0
        else:
            return self.transform(x)

    @torch.no_grad()
    def post_process(self, x: torch.Tensor) -> torch.Tensor:
        return x.clip(0, 1) * 255.0

    @torch.no_grad()
    def enhance(self, image: np.ndarray, outscale: float = 2) -> np.ndarray:
        (h, w) = image.shape[:2]
        scale_w = int((w - w % self.label_size + self.input_size) * self.scale)
        scale_h = int((h - h % self.label_size + self.input_size) * self.scale)
        # resize the input image using bicubic interpolation
        scaled = cv2.resize(image, (scale_w, scale_h), interpolation=cv2.INTER_CUBIC)
        # Preprocessing
        in_tensor = self.pre_process(scaled)  # (C, H, W)
        out_tensor = torch.zeros_like(in_tensor)  # (C, H, W)

        # slide a window from left-to-right and top-to-bottom
        for y in range(0, scale_h - self.input_size + 1, self.label_size):
            for x in range(0, scale_w - self.input_size + 1, self.label_size):
                # crop ROI from our scaled image
                crop = in_tensor[:, y : y + self.input_size, x : x + self.input_size]
                # make a prediction on the crop and store it in our output
                crop_inp = crop.unsqueeze(0).to(self.device)
                pred = self.forward(crop_inp).cpu().squeeze()
                out_tensor[
                    :,
                    y + self.pad : y + self.pad + self.label_size,
                    x + self.pad : x + self.pad + self.label_size,
                ] = pred

        out_tensor = self.post_process(out_tensor)
        output = out_tensor.permute(1, 2, 0).numpy()  # (C, H, W) to (H, W, C)
        output = output[self.pad : -self.pad * 2, self.pad : -self.pad * 2]
        output = np.clip(output, 0, 255).astype("uint8")

        # Use openCV to upsample image if scaling factor different than 2
        if outscale != 2:
            interpolation = cv2.INTER_AREA if outscale < 2 else cv2.INTER_LANCZOS4
            h, w = output.shape[0:2]
            output = cv2.resize(
                output,
                (int(w * outscale / 2), int(h * outscale / 2)),
                interpolation=interpolation,
            )

        return output, None
