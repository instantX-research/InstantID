# InstantID
<a href='https://instantid.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2401.07519'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href='https://huggingface.co/papers/2401.07519'><img src='https://img.shields.io/static/v1?label=Paper&message=Huggingface&color=orange'></a> 


**InstantID : Zero-shot Identity-Preserving Generation in Seconds**

We are currently organizing code and pre-training checkpoints, which will be available soon! Please don't hesitate to star our work.

## Abstract

There has been significant progress in personalized image synthesis with methods such as Textual Inversion, DreamBooth, and LoRA. Yet, their real-world applicability is hindered by high storage demands, lengthy fine-tuning processes, and the need for multiple reference images. Conversely, existing ID embedding-based methods, while requiring only a single forward inference, face challenges: they either necessitate extensive fine-tuning across numerous model parameters, lack compatibility with community pre-trained models, or fail to maintain high face fidelity. Addressing these limitations, we introduce InstantID, a powerful diffusion model-based solution. Our plug-and-play module adeptly handles image personalization in various styles using just a single facial image, while ensuring high fidelity.  To achieve this, we design a novel IdentityNet by imposing strong semantic and weak spatial conditions, integrating facial and landmark images with textual prompts to steer the image generation. InstantID demonstrates exceptional performance and efficiency, proving highly beneficial in real-world applications where identity preservation is paramount. Moreover, our work seamlessly integrates with popular pre-trained text-to-image diffusion models like SD1.5 and SDXL, serving as an adaptable plugin. Our codes and pre-trained checkpoints will be available at https://github.com/InstantID/InstantID.

<img src='assets/famous.png'>

## Release
- [2024/1/15] 🔥 We release the technical report.
- [2023/12/11] 🔥 We launch the project page.

## Demos

### Stylized Synthesis

<p align="center">
  <img src="assets/author.png">
</p>

### Comparison with Previous Works

<p align="center">
  <img src="assets/compare-a.png">
</p>

Comparison with existing tuning-free state-of-the-art techniques. Specifically, we compare with IP-Adapter (IPA), IP-Adapter-FaceID, and recent PhotoMaker. Among them, PhotoMaker needs to train the LoRA parameters of UNet. It can be seen that both PhotoMaker and IP-Adapter-FaceID achieves good fidelity, but there is obvious degradation of text control capabilities. In contrast, InstantID achieves better fidelity and retain good text editability (faces and styles blend better).

<p align="center">
  <img src="assets/compare-c.png">
</p>

Comparison of InstantID with pre-trained character LoRAs. We can achieve competitive results as LoRAs without any training.

<p align="center">
  <img src="assets/compare-b.png">
</p>

Comparison of InstantID with InsightFace Swapper (also known as ROOP or Refactor). However, in non-realistic style, our work is more flexible on the integration of face and background.

## Code

We are working with diffusers team and will release the code before the end of January. Starring our work will definitely speed up the process. No kidding!

## Cite
If you find InstantID useful for your research and applications, please cite us using this BibTeX:

```bibtex
@misc{wang2024instantid,
        title={InstantID: Zero-shot Identity-Preserving Generation in Seconds}, 
        author={Qixun Wang and Xu Bai and Haofan Wang and Zekui Qin and Anthony Chen},
        year={2024},
        eprint={2401.07519},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
}
