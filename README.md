# Bi-Factorial Preference Optimization: Balancing Safety-Helpfulness in Language Models

Official PyTorch implementation for our ICLR 2025 spotlight paper:

**Bi-Factorial Preference Optimization: Balancing Safety-Helpfulness in Language Models**
Authors: [Wenxuan Zhang](https://wx-zhang.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Mohamed Elhoseiny*](https://www.mohamed-elhoseiny.com/), [Adel Bibi*](www.adelbibi.com/) (* Equal Advising)

[![Paper](https://img.shields.io/badge/Paper-red?logo=arxiv&logoWidth=15)](https://arxiv.org/abs/2408.15313)
[![Jupyter Quickstart](https://img.shields.io/badge/Quickstart-orange?logo=google-colab&logoWidth=15)](https://colab.research.google.com/drive/1OpgYL_cxekAqZF8B8zuQZkPQxUIxzV0K?usp=sharing)
[![Checkpoints](https://img.shields.io/badge/🤗%20Checkpoints-grey?logoColor=white&logoWidth=20)](https://3dcompat-dataset.org/doc/dl-dataset.html)

<p align="center">
  <img src="assets/bfpo.gif" width="50%">
</p>


## 📰 News
- **(2025-01)**: Our paper is accepted to ICLR 2025 as spotlight presentation! 🎉
- **(2024-08)**: We release the paper on ArXiv. Check it out [here](https://arxiv.org/abs/2408.15313).

## 📚 Introduction
This project aims to improve the safety during the alignment of the LLMs and mitigate the potential conflicts in safety and helpfulness with low cost. We propose a novel Bi-Factorial Preference Optimization (BFPO) framework, which convert a joint RLHF reward of safety and helpfulness into a single supervised learning objective. 

This repo contains the implementation for the BFPO framework, as well as the code for the experiments in the paper.



## 🚀 Getting started
To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n bfpo python=3.10 && conda activate bfpo
```
Next, install PyTorch `v2.1.2`. We direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies as follows:

```shell
git clone https://github.com/wx-zhang/bfpo.git
cd ./bfpo
python -m pip install .
```