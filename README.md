## Welcome to our code for `TSCMamba: Mamba meets multi-view learning for time series classification`

## Overview
Our method-related code is in the folder of <b>models</b> and pre-processing related code is in the folder of <b>data_provider</b>.

## Installation
To run our code please install `PyTorch` with cuda support. For our package, we have used <b>1.13.0</b> version. It can be installed from this [link](https://pytorch.org/get-started/previous-versions/#v1130). We used CUDA 11.7 option.

Please also install Mamba. The installation procedures are mentioned [here](https://github.com/state-spaces/mamba). Please consider using a Linux system such as Ubuntu for smoother compilation. It might face some issues while installing on other systems (e.g., [issue1](https://github.com/state-spaces/mamba/issues/124), [issue2](https://github.com/state-spaces/mamba/issues/12), etc.) In addition to those mentioned above, we also used several other packages mentioned in `requirements.txt` files.

## Data Download
Download the datasets from the [official website](https://www.timeseriesclassification.com/dataset.php) in `.ts` format. Place the downloaded files in the `datasets/X` folder, where `X` is the dataset name (e.g., `SpokenArabicDigits`, `Handwriting`, etc.).
## Script running
To run for a dataset please use this command `sh ./scripts/classification/TSCMamba.sh`, for example this will run for SpokenArabicDigits dataset and will generate relevant checkpoint, result, etc. Modify it as per your requirements.

## Acknowledgements
We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositories. Their contributions have been immensely beneficial to our work.
- Mamba (https://github.com/state-spaces/mamba)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

We also thank the data creators and curators for their hard work in making these datasets publicly available.


If you find our work useful in your research, please consider citing our paper as follows:

```
@article{tscmamba,
title = {TSCMamba: Mamba meets multi-view learning for time series classification},
journal = {Information Fusion},
volume = {120},
pages = {103079},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103079},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525001526},
author = {Md Atik Ahamed and Qiang Cheng},
}
```

