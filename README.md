# Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours

## Requirements
* Python 3.5+
* CUDA 9.0
* NCCL 2.3.7

## Quick Start
* Create and activate the conda environment: 
  ```bash
  conda env create -f environment.yml
  source activate sp-nas
  ```
* Add environment variables:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/nccl-2.3.7/lib:/usr/local/cuda-9.0/lib64
  export PATH=/usr/local/cuda-9.0/bin:$PATH
  ``` 
* NAS Search [nas-search](/nas-search/): Employ NAS search 
* Train ConvNet [train-final](/train-final/): Fully train found ConvNet on ImageNet

## Citation
Please cite the Single-Path paper ([link](https://arxiv.org/abs/1904.02877)) 
in your publications if this repo helps your research:

    @inproceedings{stamoulis2019singlepath,
      author = {Stamoulis, Dimitrios and Ding, Ruizhou and Wang, Di and Lymberopoulos, Dimitrios and Priyantha, Bodhi and Liu, Jie and Marculescu, Diana}
      booktitle = {arXiv preprint arXiv:1904.02877},
      title = {Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours},
      year = {2019}
    }

