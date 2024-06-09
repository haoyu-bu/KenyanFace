# KenyanFace Mitigates Data Bia for Generative Face Models

We propose KenyanFace, a large-scale image dataset primarily composed of faces from Black individuals, and KenyanFaceHQ, a high-quality subset of 120K images at a resolution of 1024×1024.

## Dataset
The proposed KenyanFace and KenyanFaceHQ datasets are available at [here](https://portals.mdi.georgetown.edu/public/kenyanfacehq).

## Task 1: Unconditional Face Generation using Latent Diffusion Models

We use the official implementation of [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752) from [here](https://github.com/CompVis/latent-diffusion/tree/main). 

We retrain their models by incorporating the proposed KenyanFaceHQ dataset in the training data.


### Training

Download the [LDM repository](https://github.com/CompVis/latent-diffusion/tree/main) and the [taming-transformers repository](https://github.com/CompVis/taming-transformers) under *face_generation*.

Please refer to the instructions provided in the LDM repository on how to [train the LDM model from scratch](https://github.com/CompVis/latent-diffusion/tree/main?tab=readme-ov-file#train-your-own-ldms). 

We provide the config files under the directory *configs* for 
1. training the VQ-regularized autoencoder on our Combined dataset: *configs/vqgan.yaml*. Follow instructions provided in the [taming-transformers repository](https://github.com/CompVis/taming-transformers) and use the provided config file to train the VQ-regularized autoencoder. 
2. training the unconditional LDM model on our Combined dataset: *configs/ldm.yaml*. The ckpt_path in the config file should be replaced with the path of the trained VQ-regularized autoencoder.

We also share the code for loading the combined dataset under the directory *data*. Place it under *taming/data/faceshq.py*.

### Sampling

The script for sampling from unconditional LDMs can be found in the LDM [repository](https://github.com/CompVis/latent-diffusion/blob/main/scripts/sample_diffusion.py). Start it via

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r <path_of_model> -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta> 
```

### Evaluation

1. [FairFace](https://github.com/dchen236/FairFace) for Racial Classification.
2. [MaskGAN](https://github.com/switchablenorms/CelebAMask-HQ) for skin segmentation.
3. Generate [skin color](https://github.com/SonyResearch/apparent_skincolor).
4. Calculate FID scores using [torch-fidelity](https://github.com/toshas/torch-fidelity).

## Task 2: Gender Classification

The source code for gender classification experiments are located in the directory *gender_classification*.

### Requirements

   1. Please follow the [Pytorch's official documentation](https://pytorch.org/get-started/locally/) to install pytorch and torchvision.
   2. Download the [UTKFace](https://susanqq.github.io/UTKFace/), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and our proposed KenyanFace dataset.
   3. Place the cropped faces on directories: *celeba, UTKface_inthewild*, and *kenya*. Generate annotation files, with one column for image ID and one column for gender labels, and place them under the same directory as the images.

### Training
Train the model on different datasets by running the following script:

```
python train.py --dataset <DATASET_NAME> --output <OUTPUT_PATH>
```

### Evaluation

Download the [FairFace](https://github.com/joojs/fairface?tab=readme-ov-file) Dataset. Generate an annotation file of the subset of FairFace which contains equal number of faces of different races and genders by selecting images they used for service_test.

Run the evaluation script for different models.
```
python test.py --model <PATH_OF_MODEL>
```


License: CC BY 4.0
