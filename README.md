# Anodapter
Anodapter: A Unified Framework for Generating Anomaly and Mask Images Using Diffusion Models
<!-- <br> -->
[Minkyoung Shin<sup>1#</sup>](https://github.com/conel77)

<img src="anodapter_overview.png" width="800">

## Dataset
| Data and Models                          | Download                                                                                             | Place at                  |
|------------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------|
| Generated data                           | [Google Drive](https://drive.google.com/drive/folders/19uaBAdhukr1RKxI7qZyGo5cx0mBXEz4_?usp=sharing)   | $path_to_the_generated_data                          |


## Overview
Anodapter is a few-shot anomaly generation model for anomaly inspection.

The overall process is as follows:

1. During the training phase, anomaly images and corresponding masks for each class are learned simultaneously.
2. In the inference phase, an anomaly mask is first generated, followed by the generation of an anomaly image that aligns with the given mask.

## Prepare


### (1) Prepare the environment
```
Ubuntu
python 3.8
cuda==11.8
torch==2.3.0
```
### (2) Prepare MVTec dataset

Download the [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset and unzip the archive files under ```./home```.

### Dataset Structure

To train the model, please organize your dataset directory as follows:

```
<dataset_name>/
├── <class_name>/
│ ├── <anomaly_type>/
│ │ ├── images/ # Contains anomaly images
│ │ ├── masks/ # Contains corresponding anomaly masks
│ │ └── normals/ # Contains normal (non-defective) images
```

- `<dataset_name>`: Root folder of the dataset
- `<class_name>`: Object or category name
- `<anomaly_type>`: Type of anomaly within the class

Ensure that each folder is correctly populated for the training pipeline to work properly.

### Note on Object Masks

- **Object masks** (i.e., masks separating the object from the background) must be **consistent across all images within a class**.
- Object masks are generated using the **U<sup>2</sup>-Net** model.
- Since U<sup>2</sup>-Net performance may vary across images, the quality of the generated object masks may differ significantly.
- If you're using a **custom dataset**, it is recommended to run the object mask generation process **multiple times** and select the best-quality masks manually.
- The U<sup>2</sup>-Net code can be cloned from the following repository:  
  [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

> ⚠️ Although `train_anodapter.py` includes functionality to generate object masks, it runs only **once per image**, which might result in **low-quality masks**.

- For commonly used datasets **MVTec**, high-quality object masks are already provided in the `object_mask` directory.

### (3) Checkpoint for Stable-Diffusion 1.4

Download the official checkpoint of the stable diffusion model:

### (4) How to train 🔥

```bash
cd /home/Anodapter/examples/anodapter
bash train_anodapter.sh
```
We train with `MVTec` data precision on `2` NVIDIA `RTX3090` GPUs.

### (5) How to inference 🎈

```
cd /home/Anodapter/examples/anodapter
CUDA_VISIBLE_DEVICES=1 python sample_anodapter.py --size 512 --num 125 --step 100 --batch 8 --model_path $path_to_checkpoint --save_dir $path_to_the_save_dir --object_path /home/work/object_mask/$class_name --prompt_txt_path $path_to_generated_prompt_mapping.txt --adapter_path $path_to_checkpoint --anomaly_type $type_name
```

## Citation

If you make use of our work, please cite our paper:

```
@article{shin2025anodapter,
  title={Anodapter: A Unified Framework for Generating Aligned Anomaly Images and Masks Using Diffusion Models},
  author={Shin, Minkyoung and Jeong, Seonggyun and Heo, Yong Seok},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```
