import modal
import json

image = (modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("torch==2.5.1").pip_install(
    "git+https://github.com/huggingface/transformers.git").pip_install("scikit-learn").pip_install(
        "pillow==11.0.0").pip_install("scipy==1.14.1"))

# dockerfile_image = modal.Image.from_dockerfile("dockerfile")

app = modal.App("modal-test")


@app.function(image=image)
@modal.web_endpoint(method="POST")
def run_on_image_text(run_data: dict):
    from PIL import Image
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import base64
    from io import BytesIO

    texts = run_data["text_prompts"]

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    image = Image.open(BytesIO(base64.b64decode(run_data["image"])))
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

    return_dict: dict[str, list] = {"boxes": [], "conf" : [], "labels": []}
    for result in results:
        return_dict["boxes"].append(result["boxes"].tolist())
        return_dict["conf"].append(result["scores"].tolist())
        return_dict["labels"].append([texts[x] for x in result["labels"].tolist()])

    # i = 0  # Retrieve predictions for the first image for the corresponding text queries
    # text = texts[i]
    # boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    # for box, score, label in zip(boxes, scores, labels):
    #     box = [round(i, 2) for i in box.tolist()]
    #     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    return return_dict

@app.function(image=image)
@modal.web_endpoint(method="POST")
def run_on_image_image(run_data: dict):
    from PIL import Image
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import base64
    from io import BytesIO
    import numpy as np
    from torch import sigmoid
    
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    

    source_image = Image.open(BytesIO(base64.b64decode(run_data["image_source"])))
    target_image = Image.open(BytesIO(base64.b64decode(run_data["image_target"])))
    
    # Process target and source image for the model
    inputs = processor(images=target_image, query_images=source_image, return_tensors="pt")

    # Print input names and shapes
    for key, val in inputs.items():
        print(f"{key}: {val.shape}")
    
    # Get predictions
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
    
    target_sizes = torch.Tensor([target_image.size[::-1]])
    results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.98, nms_threshold=0.01, target_sizes=target_sizes)

    return_dict: dict[str, list] = {"boxes": [], "conf" : [], "labels": []}
    for result in results:
        print(result)
        return_dict["boxes"].append(result["boxes"].tolist())
        return_dict["conf"].append(result["scores"].tolist())

    # i = 0  # Retrieve predictions for the first image for the corresponding text queries
    # text = texts[i]
    # boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    # for box, score, label in zip(boxes, scores, labels):
    #     box = [round(i, 2) for i in box.tolist()]
    #     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    return return_dict