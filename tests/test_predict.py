import base64
from io import BytesIO
import os
import time

import numpy as np
from PIL import Image, ImageChops
import pytest
import requests


def local_run(model_endpoint: str, model_input: dict):
    # Maximum wait time in seconds
    max_wait_time = 1000
    # Interval between status checks in seconds
    retry_interval = 100

    total_wait_time = 0
    while total_wait_time < max_wait_time:
        response = requests.post(model_endpoint, json={"input": model_input})
        data = response.json()

        if "output" in data:
            try:
                datauri = data["output"][0]
                base64_encoded_data = datauri.split(",")[1]
                decoded_data = base64.b64decode(base64_encoded_data)
                return Image.open(BytesIO(decoded_data))
            except Exception as e:
                print("Error while processing output:")
                print("input:", model_input)
                print(data)
                raise e
        elif "detail" in data and data["detail"] == "Already running a prediction":
            print(f"Prediction in progress, waited {total_wait_time}s, waiting more...")
            time.sleep(retry_interval)
            total_wait_time += retry_interval
        else:
            print("Unexpected response data:", data)
            break
    else:
        raise Exception("Max wait time exceeded, unable to get valid response")


def image_equal_fuzzy(img_expected, img_actual, test_name="default", tol=20):
    """
    Assert that average pixel values differ by less than tol across image
    Tol determined empirically - holding everything else equal but varying seed
    generates images that vary by at least 50
    """
    img1 = np.array(img_expected, dtype=np.int32)
    img2 = np.array(img_actual, dtype=np.int32)

    mean_delta = np.mean(np.abs(img1 - img2))
    imgs_equal = mean_delta < tol
    if not imgs_equal:
        # save failures for quick inspection
        save_dir = f"/tmp/{test_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_expected.save(os.path.join(save_dir, "expected.png"))
        img_actual.save(os.path.join(save_dir, "actual.png"))
        difference = ImageChops.difference(img_expected, img_actual)
        difference.save(os.path.join(save_dir, "delta.png"))

    return imgs_equal


@pytest.fixture
def expected_image():
    return Image.open("tests/assets/out.png")


def test_seeded_prediction(expected_image):
    data = {
        "image": "https://replicate.delivery/pbxt/KIIutO7jIleskKaWebhvurgBUlHR6M6KN7KHaMMWSt4OnVrF/musk_resize.jpeg",
        "prompt": "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality",
        "scheduler": "EulerDiscreteScheduler",
        "enable_lcm": False,
        "pose_image": "https://replicate.delivery/pbxt/KJmFdQRQVDXGDVdVXftLvFrrvgOPXXRXbzIVEyExPYYOFPyF/80048a6e6586759dbcb529e74a9042ca.jpeg",
        "sdxl_weights": "protovision-xl-high-fidel",
        "pose_strength": 0.4,
        "canny_strength": 0.3,
        "depth_strength": 0.5,
        "guidance_scale": 5,
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured",
        "ip_adapter_scale": 0.8,
        "lcm_guidance_scale": 1.5,
        "num_inference_steps": 30,
        "enable_pose_controlnet": True,
        "enhance_nonface_region": True,
        "enable_canny_controlnet": False,
        "enable_depth_controlnet": False,
        "lcm_num_inference_steps": 5,
        "controlnet_conditioning_scale": 0.8,
        "seed": 1337,
    }

    actual_image = local_run("http://localhost:5000/predictions", data)
    expected_image = Image.open("tests/assets/out.png")
    test_result = image_equal_fuzzy(
        actual_image, expected_image, test_name="test_seeded_prediction"
    )
    if test_result:
        print("Test passed successfully.")
    else:
        print("Test failed.")
    assert test_result
