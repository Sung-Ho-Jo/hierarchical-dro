# Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets

This repository contains the official implementation of the paper
accepted at **ICLR 2026**.

Official implementation of the paper accepted at **ICLR 2026**.

**Authors:** Seonghyeon Kim, Sung Ho Jo, Wooseok Ha, and Minwoo Chae  
📄 [Paper (arXiv)](https://arxiv.org/abs/2510.02818)

The experiments are based on the following codebases:
- [Group DRO](https://github.com/kohpangwei/group_DRO)
- [LISA](https://github.com/huaxiuyao/LISA)

## Abstract

Conventional supervised learning methods are often vulnerable to spurious correlations, particularly under distribution shifts in test data. To address this issue, several approaches, most notably Group DRO, have been developed. While these methods are highly robust to subpopulation or group shifts, they remain vulnerable to intra-group distributional shifts, which frequently occur in minority groups with limited samples. We propose a hierarchical extension of Group DRO that addresses both inter-group and intra-group uncertainties, providing robustness to distribution shifts at multiple levels. We also introduce new benchmark settings that simulate realistic minority group distribution shifts—an important yet previously underexplored challenge in spurious correlation research. Our method demonstrates strong robustness under these conditions—where existing robust learning methods consistently fail—while also achieving superior performance on standard benchmarks. These results highlight the importance of broadening the ambiguity set to better capture both inter-group and intra-group distributional uncertainties.

## Prerequisites
- python 3.6.8
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- pillow 5.4.1
- pytorch 1.1.0
- pytorch_transformers 1.2.0
- torchvision 0.5.0a0+19315e3
- tqdm 4.32.2

## Datasets and Scripts

To execute the code, update the `root_dir` variable in `data/data.py` to reflect the directory containing your datasets. The primary entry point for running experiments is `run_expt.py`. Below are example commands for running the code with each dataset.

We conducted experiments on three datasets: CMNIST, Waterbirds, and CelebA. For each dataset, we also evaluate scenarios involving minority group distribution shifts (`shift = True`). For the Waterbirds dataset, we additionally explore the case of a corrected version where mislabeling in the original data has been addressed.


### CMNIST
This dataset is built using MNIST and will be automatically downloaded when you execute the following script.

The command to run our method is as follows:
```
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.01 --batch_size 128 --weight_decay 0.01 --model resnet50 --n_epochs 50 --reweight_groups --robust --generalization_adjustment 1 --epsilon 72/255 --scheduler
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.01 --batch_size 128 --weight_decay 0.01 --model resnet50 --n_epochs 50 --reweight_groups --robust --generalization_adjustment 1 --epsilon 72/255 --scheduler --shift
```

### Waterbirds
This dataset can be accessed through the link provided in the [group_DRO](https://github.com/kohpangwei/group_DRO) repository.
The repository includes details about the necessary files and their respective download links. It also specifies the folder structure where these files should be saved. For example, the waterbird dataset files need to be organized under a structured folder (e.g., `cub/data/waterbird_complete95_forest2water2`) to ensure proper integration with the code.
The command to run our method is as follows:

```
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.00001 --batch_size 128 --weight_decay 1.0 --model resnet50 --n_epochs 300 --reweight_groups --robust --generalization_adjustment 2 --epsilon 12/255
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.00001 --batch_size 128 --weight_decay 1.0 --model resnet50 --n_epochs 300 --reweight_groups --robust --generalization_adjustment 0 --epsilon 36/255 --edited_mislabel
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.00001 --batch_size 128 --weight_decay 1.0 --model resnet50 --n_epochs 300 --reweight_groups --robust --generalization_adjustment 2 --epsilon 12/255 --shift
```

### CelebA
This dataset can be accessed through the link provided in the [group_DRO](https://github.com/kohpangwei/group_DRO) repository.
The repository contains detailed instructions on downloading the required files, such as img_align_celeba.zip, and outlines how to organize them into the appropriate folder structure (e.g., `celebA/data`) for seamless integration with the code.
The command to run our method is as follows:
```
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.00001 --batch_size 128 --weight_decay 0.01 --model resnet50 --n_epochs 30 --reweight_groups --robust --generalization_adjustment 1 --epsilon 12/255
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.00001 --batch_size 128 --weight_decay 0.01 --model resnet50 --n_epochs 30 --reweight_groups --robust --generalization_adjustment 1 --epsilon 84/255 --shift
```












