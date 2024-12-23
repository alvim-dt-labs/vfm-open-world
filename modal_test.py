import modal
import json

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1")
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .pip_install("scikit-learn")
    .pip_install("pillow==11.0.0")
    .pip_install("scipy==1.14.1")
)

# dockerfile_image = modal.Image.from_dockerfile("dockerfile")

app = modal.App("modal-test")
@app.function(image=image)
@modal.web_endpoint(method="POST")
def run_on_image(run_data: dict):
    from PIL import Image
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import base64
    from io import BytesIO

    texts = run_data["text_prompt"]

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    image = Image.open(BytesIO(base64.b64decode(run_data["image"])))
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    return { "bboxes": box}