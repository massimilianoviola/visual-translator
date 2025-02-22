import gradio as gr
import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import warnings
from huggingface_hub import login
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

login()

warnings.filterwarnings("ignore")
keras.config.set_floatx("bfloat16")

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def read_image(path, target_size):
    image = Image.open(path)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def parse_bbox_and_labels(detokenized_output: str):
    matches = re.finditer(
        '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        ' (?P<label>.+?)( ;|$)',
        detokenized_output,
    )
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
        labels.append(d['label'])
    return np.array(boxes), np.array(labels)

def display_boxes(image, boxes, labels, target_image_size):
    h, l = target_image_size
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    for i in range(boxes.shape[0]):
        y, x, y2, x2 = (boxes[i]*h)
        width = x2 - x
        height = y2 - y
        rect = patches.Rectangle((x, y),
                                 width,
                                 height,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        plt.text(x, y, labels[i], color='red', fontsize=12)
        ax.add_patch(rect)
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return Image.fromarray(img_array)

def detect(image_path):
    target_size = (224, 224)
    raw_image = read_image(image_path, target_size)

    prompt = "Identify the object at the tip of the arrow, ignoring the arrow itself;"

    inputs = processor(prompt, raw_image, padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    detected_object = processor.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

    prompt_object = f"detect the {detected_object};"

    inputs = processor(prompt_object, raw_image, padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    input_string = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    boxes, labels = parse_bbox_and_labels(input_string)

    return display_boxes(raw_image, boxes, labels, target_size)

# detect() returns the image as a PIL.Image object
img = detect("./IMG_0181_arrow.jpg")
img.show()

default_images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
]

demo = gr.Interface(
    fn=detect,  # Call detect(image_path) directly when an image is uploaded
    inputs=gr.File(file_count="single", type="file", label="Upload an image"),
    outputs=gr.Image(label="Detected Objects"),  # Output the image with detections
    title="Visual Translator",
    description="Upload an image (drag-and-drop or file selector), and the model will detect and describe objects.",
    examples=default_images
)

demo.launch()