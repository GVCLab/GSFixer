## GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors
## [Arxiv](https://www.arxiv.org/pdf/2508.09667)

<table class="center">
    <tr style="font-weight: bolder;">
        <td>&emsp;&emsp;3DGS Artifact&emsp;&emsp;&emsp; Difix3D+&emsp;&emsp;&emsp;&emsp; GenFusion&emsp;&emsp;&emsp;GSFixer (Ours)&emsp;&emsp;&emsp;&emsp; GT</td>
    </tr>
  <tr>
  <td>
    <img src=assets/comparison1.gif style="width: 100%; height: auto;">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/comparison2.gif style="width: 100%; height: auto;">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/comparison3.gif style="width: 100%; height: auto;">
  </td>
  </tr>
</table>




## Introduction

We are currently cleaning up the code and awaiting company approval. The code, checkpoints, and data will be released as soon as possible.

TL;DR: using 2D semantic (DINOv2) and 3D geometric (VGGT) features of reference views (sparse input views) to guide the video diffusion (CogVideoX) process, enabling semantic and 3D consistency in restoring artifact novel views.

Motivation: Recent approaches have sought to leverage generative priors to complete information for under-constrained regions; they struggle to generate content that remains consistent with input observation. To address this challenge, considering the artifacts finally lie in the 2D image space and are caused by suboptimal 3DGS representations in 3D space, we propose injecting both 2D semantic and 3D geometric control signals of reference views to guide the video diffusion process, enabling both semantic and 3D consistency in restorating the artifact's novel views.

## Tesear
![Tesear](assets/tesear.png)

## Pipeline 
![Pipeline](assets/gsfixer_framework.png)


## Related Works
Including but not limited to: [CogVideoX](https://github.com/zai-org/CogVideo), [VGGT](https://github.com/facebookresearch/vggt), [DINOv2](https://github.com/facebookresearch/dinov2), [Difix3D+](https://github.com/nv-tlabs/Difix3D), [GenFusion](https://github.com/Inception3D/GenFusion), [3DGS-Enhancer](https://github.com/xiliu8006/3DGS-Enhancer), [ReconX](https://github.com/liuff19/ReconX), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster)...

## Citation

If you find the work useful, please consider citing:
```
@article{yin2025gsfixer,
  title={GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors},
  author={Yin, Xingyilang and Zhang, Qi and Chang, Jiahao and Feng, Ying and Fan, Qingnan and Yang, Xi and Pun, Chi-Man and Zhang, Huaqi and Cun, Xiaodong},
  journal={arXiv preprint arXiv:2508.09667},
  year={2025}
}
```