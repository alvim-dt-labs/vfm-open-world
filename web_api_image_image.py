import requests
import argparse
import json
import base64

# parser = argparse.ArgumentParser()
# parser.add_argument("-s",
#                     "--image-source",
#                     action="store",
#                     help="Path of image to be used as query.",
#                     required=True)
# parser.add_argument("-t", "--image-target", nargs="+", help="Target image to be queried against.", required=True)
# args = parser.parse_args()

url = 'https://strifezeek--modal-test-run-on-image-image-dev.modal.run'
# headers = {f"Authorization": "Bearer " + args.token}
image_target_b64 = base64.b64encode(open("./data/images/giraffe-zebra.jpg", "rb").read()).decode()
image_query_b64 = base64.b64encode(open("./data/images/zebra.jpg", "rb").read()).decode()
response = requests.post(url, json={"image_source": image_query_b64, "image_target": image_target_b64})

print(response.text)

temp = {
    "boxes": [[[2634.992431640625, 548.7421264648438, 3996.7412109375, 660.3448486328125],
               [5045.21728515625, 2091.459716796875, 5319.68798828125, 2354.028564453125],
               [4551.08544921875, 2874.89599609375, 5810.37939453125, 2920.237060546875],
               [-688.3536987304688, 2900.536376953125, 4166.41259765625, 2927.43994140625],
               [76.98806762695312, 52.505401611328125, 5191.74462890625, 3863.871337890625]]],
    "conf": [[0.9843193888664246, 0.9881054759025574, 0.9801443815231323, 0.9970890879631042, 0.9999988675117493]],
    "labels": []
}
