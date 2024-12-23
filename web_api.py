import requests
import argparse
import json
import base64

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--image-path", action="store", help="Path of iamge to be analysed by the network.", required=True)
# parser.add_argument("-t", "--text-prompt", action="store", help="Text prompt", required=True)
# args = parser.parse_args()

url = 'https://strifezeek--modal-test-run-on-image-dev.modal.run'
# headers = {f"Authorization": "Bearer " + args.token}
image_b64 = base64.b64encode(open("./data/images/2-dogs.jpeg", "rb").read()).decode()
response = requests.post(url, json={
    "text_prompt" : "detect dogs",
    "image" : image_b64
    
})

print(response.text)