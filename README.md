# Toy Implementation of Attention Mechanisms 🤖



Welcome to the Toy Implementation of Attention Mechanisms! This repository is designed to help beginners grasp the core concepts of attention mechanisms used in transformer models. By following this guide, you'll understand how single-head attention works, with clear visualizations and step-by-step explanations. Let's dive in!

## Overview
This project provides a simple implementation of the operations involved in a single-head attention mechanism. The goal is to demystify how attention works in transformer models, making it easier for you to learn and experiment.



## Features
- Clear Explanations: Detailed comments and descriptions to help you understand each step.
- Visualizations: Plotting utilities to visualize key matrices and operations, making complex concepts easier to grasp.
- Hands-on Code: Ready-to-use code snippets to run and experiment with.

## Installation
To get started, clone the repository and install the necessary dependencies:
```
git clone https://github.com/IParraMartin/attention-mechanisms.git
cd attention-mechanisms
pip install -r requirements.txt
```

## Usage
Here's a quick guide on how to use the code in this repository.

### 🛠️ Main Parameters
We start by defining the main parameters and dimensions used in the implementation. The base model of Vaswani et al. (2017) uses 512 dimensions for the model (embedding) and parallelizes computations in 8 heads, resulting in 64 dimensions per head.
```
d_model = 512
heads = 8
dim_k = d_model // heads
dim_v = d_model // heads
```

### 🧸 "Toy" Embeddings
We create random embedding vectors to simulate a sequence of 128 tokens.
```
n_embeddings = 128
embeddings = torch.randn(n_embeddings, dim_k)
```

## 👁️ Visualization Tools
To make learning easier, we use plotting utilities to visualize matrices at each key step. These visualizations help in understanding the data transformations happening at each stage of the attention mechanism. The functions can be found in ```utils.py```

## Some Final Remarks
This toy implementation provides a hands-on approach to learning about attention mechanisms in transformer models. Feel free to experiment with the code, modify parameters, and visualize the results to deepen your understanding.

Happy learning! 👨🏽‍🎓
