<p align="center">
  <img src="ensemble_logo.jpg" alt="Logo" width="400">
  <br /> <br / >
</p>

# NdLinear: The next-gen replacement for the linear layer
A drop-in pytorch module to replace the standard linear layer in neural networks by [Ensemble AI](https://ensemblecore.ai).


**Ensemble Platform**  
Want smaller, faster models **without** accuracy loss? Tried pruning or quantization and hit a wall?

👉 [Try the full Ensemble platform →](https://app.ensemblecore.ai/signup)  _(10M free credits on signup)_

---

## NdLinear

A PyTorch module that replaces the standard linear layer in neural networks.

- ✅ Plug-and-play replacement for `nn.Linear`
- 📦 Lightweight, parameter-efficient
- 🧠 Preserves multivariate structure natively

> **Example**: A 130M parameter DiT model using NdLinear outperformed a 457M baseline on the FID benchmark for ImageNet100

---

## Ensemble Platform (powered by NdLinear)

Upload any model – get back a smaller, faster version.  
No accuracy loss.

- 🔁 Automatically swaps layers and tunes hyperparams
- 📉 Shrinks model size (parameter count) by up to 8x
- 🛠 Tailor uploaded models to your **hardware & finetuning constraints**
- 🧰 Export to ONNX, TensorRT, SNPE, and more
- 💡 Designed to work alongside other compression techniques(pruning, quantization, distillation) 
- 🎁 Includes **10M free credits** on signup

👉 [Try the Ensemble Platform for free→](https://app.ensemblecore.ai/signup)  
📺 [Or see a demo →](https://ensemblecore.ai)

---

## 🧬 Technical Overview

NdLinear preserves the multi-dimensional structure of data, enhancing representational power with fewer parameters.  
Rather than flattening tensors, it transforms them across a structured set of vector spaces—capturing dependencies standard fully connected layers discard.

👉 [Link to paper→](https://arxiv.org/pdf/2503.17353)


## Key Features

- **Structure Preservation:** Retains the original data format and shape.
- **Parameter Efficiency:** Reduces parameter count while improving performance.
- **Minimal Overhead:** Maintains the same complexity as conventional linear layers.
- **Flexible Integration:** Seamlessly replaces existing linear layers.

## Installation

To integrate NdLinear into your projects, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ensemble-core/NdLinear.git
cd NdLinear
pip install . 
```

Alternatively, if packaged, install via pip:

```bash
pip install ndlinear
```
Or, via conda:

```bash
conda install conda-forge::ndlinear
```

## Usage

NdLinear can be utilized in various neural network architectures such as CNNs, RNNs, and Transformers.

### Example 1: Replacing a Standard Linear Layer with NdLinear

```python
import torch
from ndlinear import NdLinear

input_tensor = torch.randn(32, 28, 28, 3)  # Batch of images

ndlinear_layer = NdLinear(input_dims=(28, 28, 3), hidden_size=(64, 64, 6))

output = ndlinear_layer(input_tensor)
```

### Example 2: Transformer

In transformer architectures, you might need to manipulate multi-dimensional tensors for efficient linear operations. Here's how you can use `NdLinear` with a 3D input tensor:

```python
import torch 
from ndlinear import NdLinear

input_tensor = torch.randn(32, 28, 28) # Input with shape : (batch_size, num_tokens, token_dim)

# Reshape the input tensor for linear operations
input_tensor = input_tensor.reshape(-1, 28, 1)  # New shape: (batch_size * num_tokens, token_dim, 1)

# Define an NdLinear layer with suitable input and hidden dimensions
ndlinear_layer = NdLinear(input_dims=(28, 1), hidden_size=(32, 1))

# Perform the linear transformation
output = ndlinear_layer(input_tensor)

# Reshape back to the original dimensions after processing
output = output.reshape(32, 28, -1)  # Final output shape: (32, 28, 32)
```

This example illustrates how `NdLinear` can be integrated into transformer models by manipulating the tensor shape, thereby maintaining the structure necessary for further processing and achieving efficient projection capabilities.

### Example 3: Multilayer Perceptron 

This example demonstrates how to use the `NdLinear` layers in a forward pass setup, making integration into existing MLP structures simple and efficient.

```python 
import torch
from ndlinear import NdLinear

input_tensor = torch.randn(32, 128)

# Define the first NdLinear layer for the MLP with input dimensions (128, 8) and hidden size (64, 8)
layer1 = NdLinear(input_dims=(128, 8), hidden_size=(64, 8))

# Define the second NdLinear layer for the MLP with input dimensions (64, 8) and hidden size (10, 2)
layer2 = NdLinear(input_dims=(64, 8), hidden_size=(10, 2))

x = F.relu(layer1(input_tensor))

output = layer2(x)
```

### Example 4: Edge Case

When `input_dims` and `hidden_size` are one-dimensional, `NdLinear` functions as a conventional `nn.Linear` layer, serving as an edge case where `n=1`.

```python
from ndlinear import NdLinear

# Defining NdLinear with one-dimensional input and hidden sizes
layer1 = NdLinear(input_dims=(32,), hidden_size=(64,))
```

## NdLinearGated

<img src="NdLinearGated.png" alt="NdLinearGated" width="100">

NdLinearGated extends the core functionality of NdLinear by incorporating sophisticated gating mechanisms to control information flow. This allows models to selectively transform input dimensions, enhancing representational power and efficiency.

### Key Features

- **Selective Information Flow:** Dynamic gating mechanisms that control which transformations are applied
- **Multiple Gating Modes:** Support for soft (continuous) and hard (binary) gating approaches
- **Dimension Selection:** Apply gating to all dimensions, only the first dimension, or automatically to the most important dimensions

### Usage

NdLinearGated can be integrated into neural networks with fine-grained control over the gating behavior:

```python
import torch
from ndlinear import NdLinearGated

# Create input tensor
input_tensor = torch.randn(32, 28, 28, 3)  # Batch of images

# Initialize NdLinearGated with soft gating on all dimensions
gated_layer = NdLinearGated(
    input_dims=(28, 28, 3),
    hidden_size=(64, 64, 6),
    gating_mode="soft",
    gated_modes="all"
)

# Forward pass with gating
output = gated_layer(input_tensor)
```

### Gating Configurations

NdLinearGated offers various configurations to suit different modeling needs:

```python
# Apply hard gating only to the first dimension
first_dim_gated = NdLinearGated(
    input_dims=(128, 8),
    hidden_size=(64, 8),
    gating_mode="hard",
    gated_modes="first"
)

# Apply soft gating to top-k dimensions with highest standard deviation
topk_gated = NdLinearGated(
    input_dims=(28, 28, 3),
    hidden_size=(64, 64, 6),
    gating_mode="soft",
    gated_modes="topk"
)
```

### Recommended Configuration

Based on extensive experimentation, we recommend the following configuration for optimal performance:

```python
optimal_gated = NdLinearGated(
    input_dims=(28, 28, 3),
    hidden_size=(64, 64, 6),
    gating_mode="soft",
    gated_modes="topk",
    gating_hidden_dim=16  # Adjust based on your model size
)
```

This configuration (soft gating + top-k modes) consistently delivers:

- **Higher Accuracy:** Improves performance by ~0.5-0.7% over baseline NdLinear
- **Compute Efficiency:** Reduces computational load by 50-75% by activating only the most useful projections
- **Training Stability:** Shows stable training and smooth gate entropy decay
- **Enhanced Interpretability:** Provides clearer insights into which dimensions are most important

The soft gating mechanism offers better stability compared to hard gating, while top-k mode selection focuses computational resources on the most informative dimensions. The gating_hidden_dim parameter can be adjusted based on your specific model architecture and dataset requirements.

## Examples of Applications

NdLinear is versatile and can be used in:

- **Image Classification:** Run `cnn_img_classification.py`.
- **Time Series Forecasting:** Use `ts_forecast.py`.
- **Text Classification:** Launch `txt_classify_bert.py`.
- **Vision Transformers:** Execute `vit_distill.py`.


## Community Engagement

Join the community and enhance your projects using NdLinear in Hugging Face, Kaggle, and GitHub.

Join our Discord! https://discord.gg/6DWHusWN

[//]: # (## Citation)

[//]: # ()
[//]: # (If you find NdLinear useful in your research, please cite our work:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{reneau2025ndlinear,)

[//]: # (  title={NdLinear Is All You Need for Representation Learning},)

[//]: # (  author={Reneau, Alex and Hu, Jerry Yao-Chieh and Zhuang, Zhongfang and Liu, Ting-Chun},)

[//]: # (  journal={Ensemble AI},)

[//]: # (  year={2025},)

[//]: # (  note={\url{https://arxiv.org/abs/2503.17353}})

[//]: # (})

[//]: # (```)

## Contact

For questions or collaborations, please contact [Alex Reneau](mailto:alex@ensemblecore.ai).

## License

This project is distributed under the Apache 2.0 license. View the [LICENSE](https://github.com/ensemble-core/NdLinear/blob/main/LICENSE) file for more details.
