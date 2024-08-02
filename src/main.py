import torch
import cv2
from PIL import Image
import numpy as np

from src.groundingdino.service import GroundingDino
from src.lama.service import process_inpaint
from src.sam.service import SAM


class Service:
    def __init__(self) -> None:
        self.grounding_dino = GroundingDino()
        self.sam = SAM()

    def run(self, image_dir, text_prompt):
        if not isinstance(image_dir, Image.Image):
            # img_input = cv2.imread(image_dir)
            image_pil = Image.open(image_dir)
        else:
            image_pil = image_dir

        box_threshold = 0.3
        text_threshold = 0.25

        boxes, logits, phrases = self.grounding_dino.predict_dino(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.sam.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)

        im_mask = np.zeros((masks.shape[1], masks.shape[2], 4))

        background = np.where(
            (masks[0, :, :] == 0) & (masks[0, :, :] == 0) & (masks[0, :, :] == 0)
        )
        drawing = np.where(
            (masks[0, :, :] == 1) & (masks[0, :, :] == 1) & (masks[0, :, :] == 1)
        )
        im_mask[background] = [0, 0, 0, 255]
        im_mask[drawing] = [0, 0, 0, 0]  # RGBA

        # Define the structuring element (kernel) for dilation
        kernel_size = 31  # Adjust the size as needed
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Perform dilation
        dilated_mask = cv2.erode(im_mask, kernel, iterations=1)
        # LAMA
        output = process_inpaint(np.array(image_pil), np.array(dilated_mask))

        return output
