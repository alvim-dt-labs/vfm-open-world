from PIL import Image
import cv2
import numpy as np
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import base64
from io import BytesIO

run_data = {"text_prompts": ["lion", "zebra" ,"giraffe"], "image" : base64.b64encode(open("./data/images/giraffe-zebra.jpg", "rb").read()).decode()}
texts = run_data["text_prompts"]

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

image = Image.open(BytesIO(base64.b64decode(run_data["image"])))
inputs = processor(text=texts, images=image, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

return_dict: dict[str, list] = {"boxes": [], "scores" : [], "labels": []}
for result in results:
    print(result)
    return_dict["boxes"].append(result["boxes"].tolist())
    return_dict["scores"].append(result["scores"].tolist())
    return_dict["labels"].append([texts[x] for x in result["labels"].tolist()])

# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = texts[i]
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
print(return_dict)

from PIL import ImageDraw

i = 0
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.numpy().astype(np.int32).tolist()]
    x1, y1, x2, y2 = tuple(box)
    img = cv2.rectangle(img, (x1,y1), (x2,y2), [200,0,0], 5)
    cv2.putText(img, texts[label], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,200], 2)

cv2.imshow("image", img)
cv2.waitKey(0)
