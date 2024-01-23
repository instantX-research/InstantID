# InstantID Cog Model

[![Replicate](https://replicate.com/zsxkib/instant-id/badge)](https://replicate.com/zsxkib/instant-id)

## Overview
This repository contains the implementation of [InstantID](https://github.com/InstantID/InstantID) as a [Cog](https://github.com/replicate/cog) model. 

Using [Cog](https://github.com/replicate/cog) allows any users with a GPU to run the model locally easily, without the hassle of downloading weights, installing libraries, or managing CUDA versions. Everything just works.

## Development
To push your own fork of InstantID to [Replicate](https://replicate.com), follow the [Model Pushing Guide](https://replicate.com/docs/guides/push-a-model).

## Basic Usage
To make predictions using the model, execute the following command from the root of this project:

```bash
cog predict \
-i image=@examples/sam_resize.png \
-i prompt="analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality" \
-i negative_prompt="nsfw" \
-i width=680 \
-i height=680 \
-i ip_adapter_scale=0.8 \
-i controlnet_conditioning_scale=0.8 \
-i num_inference_steps=30 \
-i guidance_scale=5
```

<table>
  <tr>
    <td>
      <p align="center">Input</p>
      <img src="https://replicate.delivery/pbxt/KGy0R72cMwriR9EnCLu6hgVkQNd60mY01mDZAQqcUic9rVw4/musk_resize.jpeg" alt="Sample Input Image" width="90%"/>
    </td>
    <td>
      <p align="center">Output</p>
      <img src="https://replicate.delivery/pbxt/oGOxXELcLcpaMBeIeffwdxKZAkuzwOzzoxKadjhV8YgQWk8IB/result.jpg" alt="Sample Output Image" width="100%"/>
    </td>
  </tr>
</table>

## Input Parameters

The following table provides details about each input parameter for the `predict` function:

| Parameter                       | Description                        | Default Value                                                                                                  | Range       |
| ------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------- |
| `image`                         | Input image                        | A path to the input image file                                                                                 | Path string |
| `prompt`                        | Input prompt                       | "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, ... " | String      |
| `negative_prompt`               | Input Negative Prompt              | (empty string)                                                                                                 | String      |
| `width`                         | Width of output image              | 640                                                                                                            | 512 - 2048  |
| `height`                        | Height of output image             | 640                                                                                                            | 512 - 2048  |
| `ip_adapter_scale`              | Scale for IP adapter               | 0.8                                                                                                            | 0.0 - 1.0   |
| `controlnet_conditioning_scale` | Scale for ControlNet conditioning  | 0.8                                                                                                            | 0.0 - 1.0   |
| `num_inference_steps`           | Number of denoising steps          | 30                                                                                                             | 1 - 500     |
| `guidance_scale`                | Scale for classifier-free guidance | 5                                                                                                              | 1 - 50      |

This table provides a quick reference to understand and modify the inputs for generating predictions using the model.


