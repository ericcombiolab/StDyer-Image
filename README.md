
---

<div align="center">

# stDyer-image

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description

stDyer-image is a clustering analysis method for sptailly resolved transcriptomic/proteomics data.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ericcombiolab/stDyer-image.git
cd stDyer-image

# create conda environment
conda env create -f stdyer_image.yml
conda activate stdyer_image
```

## Tutorial
There is a tutorial notebook [tutorial.ipynb](tutorial.ipynb) that demonstrates how to train the model with a single slice dataset. For more advanced usage using command line, please refer to the following sections:

### For the dataset with a single slice
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=example.yaml
```

The predicted cluster labels will be saved to anndata(.h5ad) files in logs/logger_logs folder. The raw predicted cluster labels is in adata.obs["pred_labels"].


The detected spatially variable genes will be saved in adata.uns["svg_dict"].

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20
```

### For the large dataset (multiple GPUs)
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/) with multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py experiment=example_ddp.yaml trainer.devices=2
```

To train model with your own dataset, you can copy the [configs/experiment/example_ddp.yaml](configs/experiment/example_ddp.yaml) to [configs/experiment/your_experiment.yaml](configs/experiment/your_experiment.yaml) file and modify it to your needs. The required data format is h5ad, which can be created by [AnnData](https://anndata.readthedocs.io/en/latest/). The "spatial" key in the obsm attribute of the anndata object (`adata.obsm["spatial"]`) indicates spatial coordinates and is necessary for constructing spatial adjacency graph. The full path to h5ad file is `data_dir/dataset_dir/data_file_name`. You can also specify the requred number of clusters with the parameter `num_classes` in your_experiment.yaml as well. The config file has rich comments for explaining the parameters.

```bash
cp configs/experiment/example_ddp.yaml configs/experiment/your_experiment.yaml
python run.py experiment=your_experiment.yaml
```
