# backend_segmentation.py
#
# Florence-2 inference + post-processing utilities

import io
import json
import re
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import AutoProcessor, AutoModelForCausalLM

# MODEL_ID = "microsoft/Florence-2-large-ft"
DEVICE = torch.device("cpu")  # Force CPU usage only

# ────────────────────────── load model once ──────────────────────────
# _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# _model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, trust_remote_code=True
# ).to(DEVICE).eval()

MODEL_ID = 'microsoft/Florence-2-large' # , use_safetensors=True
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_safetensors=True)


############ define the prediction function ##################
def generate_segmentation(image, text_input, task_prompt):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    # import pdb; pdb.set_trace()
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        # image_size=(image.width, image.height)
        image_size=(image.shape[1], image.shape[0])
    )

    return parsed_answer



from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image
    # import pdb; pdb.set_trace()
    draw = ImageDraw.Draw(image)


    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    # Save or display the image
    #image.show()  # Display the image
    # display(image)
