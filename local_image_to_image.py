from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import base64
from io import BytesIO
import numpy as np
from scipy.special import expit as sigmoid
import cv2
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

run_data = {
    "image_source" : base64.b64encode(open("./data/images/zebra.jpg", "rb").read()).decode(),
    "image_target" : base64.b64encode(open("./data/images/giraffe-zebra.jpg", "rb").read()).decode() ,
} 


source_image = Image.open(BytesIO(base64.b64decode(run_data["image_source"])))
target_image = Image.open(BytesIO(base64.b64decode(run_data["image_target"])))

target_sizes = torch.Tensor([target_image.size[::-1]])

# Process source image
source_pixel_values = processor(images=source_image, return_tensors="pt").pixel_values

# For visualization, we need the preprocessed source image (i.e. padded and resized, but not yet normalized)
unnormalized_source_image = get_preprocessed_image(source_pixel_values)

# Get image features
with torch.no_grad():
  feature_map = model.image_embedder(source_pixel_values)[0]
print(feature_map.shape)

# Rearrange feature map
batch_size, height, width, hidden_size = feature_map.shape
image_features = feature_map.reshape(batch_size, height * width, hidden_size)

# Get objectness logits
objectnesses = model.objectness_predictor(image_features)
print(objectnesses)

num_patches = (model.config.vision_config.image_size // model.config.vision_config.patch_size)**2
print(num_patches)

source_boxes = model.box_predictor(image_features, feature_map=feature_map)
source_class_embeddings = model.class_predictor(image_features)[1]

# Remove batch dimension
objectnesses = np.array(objectnesses[0].detach())
source_boxes = np.array(source_boxes[0].detach())
source_class_embeddings = np.array(source_class_embeddings[0].detach())

# Let's show the top 3 patches
top_k = 1
objectnesses = sigmoid(objectnesses)
objectness_threshold = np.partition(objectnesses, -top_k)[-top_k]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(unnormalized_source_image, extent=(0, 1, 1, 0))
ax.set_axis_off()

for i, (box, objectness) in enumerate(zip(source_boxes, objectnesses)):
  if objectness < objectness_threshold:
    continue

  cx, cy, w, h = box
  ax.plot(
      [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
      [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
      color='lime',
  )

  print("Index:", i)
  print("Objectness:", objectness)

  ax.text(
      cx - w / 2 + 0.015,
      cy + h / 2 - 0.015,
      f'Index {i}: {objectness:1.2f}',
      ha='left',
      va='bottom',
      color='black',
      bbox={
          'facecolor': 'white',
          'edgecolor': 'lime',
          'boxstyle': 'square,pad=.3',
      },
  )

ax.set_xlim(0, 1)
ax.set_ylim(1, 0)
ax.set_title(f'Top {top_k} objects by objectness')

# fig.show()
# plt.waitforbuttonpress()


query_object_index = i  # Index of the cat box above.
query_embedding = source_class_embeddings[query_object_index]

from transformers.image_transforms import center_to_corners_format

img_w, img_h = unnormalized_source_image.size

# crop out cat from source image
box = source_boxes[query_object_index]
# convert from center_x, center_y, width, height to x1, x2, y1, y2
box = center_to_corners_format(box)
# rescale boxes to size of the source image
scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
boxes = torch.tensor(box) * scale_fct
x1, y1, x2, y2 = tuple(boxes.tolist())
cropped_image = unnormalized_source_image.crop((x1, y1, x2, y2))


# Process target image
target_pixel_values = processor(images=target_image, return_tensors="pt").pixel_values
unnormalized_target_image = get_preprocessed_image(target_pixel_values)

with torch.no_grad():
  feature_map = model.image_embedder(target_pixel_values)[0]

# Get boxes and class embeddings (the latter conditioned on query embedding)
b, h, w, d = feature_map.shape
target_boxes = model.box_predictor(
    feature_map.reshape(b, h * w, d), feature_map=feature_map
)

target_class_predictions = model.class_predictor(
    feature_map.reshape(b, h * w, d),
    torch.tensor(query_embedding[None, None, ...]),  # [batch, queries, d]
)[0]


# Remove batch dimension and convert to numpy:
target_boxes = np.array(target_boxes[0].detach())
target_logits = np.array(target_class_predictions[0].detach())

# Take the highest scoring logit
top_ind = np.argmax(target_logits[:, 0], axis=0)
score = sigmoid(target_logits[top_ind, 0])


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(unnormalized_target_image, extent=(0, 1, 1, 0))
ax.set_axis_off()

# Get the corresponding bounding box
cx, cy, w, h = target_boxes[top_ind]
ax.plot(
    [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
    [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
    color='lime',
)

ax.text(
    cx - w / 2 + 0.015,
    cy + h / 2 - 0.015,
    f'Score: {score:1.2f}',
    ha='left',
    va='bottom',
    color='black',
    bbox={
        'facecolor': 'white',
        'edgecolor': 'lime',
        'boxstyle': 'square,pad=.3',
    },
)

ax.set_xlim(0, 1)
ax.set_ylim(1, 0)
ax.set_title(f'Closest match')

fig.show()
plt.waitforbuttonpress()





cv2.imshow("image", img)
cv2.waitKey(0)

# return_dict: dict[str, list] = {"boxes": [], "conf" : [], "labels": []}
# for result in results:
#     print(result)
#     return_dict["boxes"].append(result["boxes"].tolist())
#     return_dict["conf"].append(result["scores"].tolist())

# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = texts[i]
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")