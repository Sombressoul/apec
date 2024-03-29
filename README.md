# APEC and MAPEC Activation Functions for Neural Networks

- [APEC and MAPEC Activation Functions for Neural Networks](#apec-and-mapec-activation-functions-for-neural-networks)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Activation Functions](#activation-functions)
  - [Mathematical Formulation](#mathematical-formulation)
    - [APEC (Asymmetric Parametric Exponential Curvature)](#apec-asymmetric-parametric-exponential-curvature)
    - [MAPEC (Multiplicative Asymmetric Parametric Exponential Curvature)](#mapec-multiplicative-asymmetric-parametric-exponential-curvature)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Overview
This repository introduces two novel activation functions, APEC (Asymmetric Parametric Exponential Curvature) and its variant MAPEC (Multiplicative APEC), designed for deep learning models to capture complex patterns with improved performance. Functions have been tested on the CIFAR-100 dataset (results included) and on some of my experimental models (results not included).

## Installation
```bash
pip install apec-afn
```

## Usage
```python
import torch
from apec import MAPEC

x = torch.randn([8])
f = MAPEC()

print(f(x))
```

## Activation Functions
- **APEC**: Offers a balance between flexibility and performance, as demonstrated by the improvement over traditional functions on CIFAR-100.
- **MAPEC**: An extension of APEC with an additional multiplicative term, allowing for an even richer model expressiveness and an observed faster convergence (up to 15%).

## Mathematical Formulation

### APEC (Asymmetric Parametric Exponential Curvature)
![APEC](doc/APEC_fn_plot.png)

APEC is designed to introduce a non-linear response with an adjustable curvature, defined by:
$$f(x) = \alpha + \frac{\beta - x}{(\gamma - \exp(-x)) + \epsilon}$$

- **Initialization**: Parameters `a` and `b` are initialized with a normal distribution of zero mean and a standard deviation of 0.35. Parameter `g` is initialized with a mean of -1.375 and a standard deviation of 0.35.
- **Constraints**: The default constraints for `a`, `b`, and `g` are [-2.0, +2.0], [-2.5, +2.5], and [-2.5, -0.25], respectively.
- **Stability**: A small constant `eps` (1.0e-5) is added to prevent division by zero.

### MAPEC (Multiplicative Asymmetric Parametric Exponential Curvature)
![MAPEC](doc/MAPEC_fn_plot.png)

MAPEC extends APEC by adding a multiplicative term, enhancing its flexibility:
$$f(x) = (\alpha + \frac{\beta - x}{-abs(\gamma) - \exp(-x) - \epsilon} + (x \cdot \delta)) \cdot \zeta$$

- **Initialization**: Parameters initialization values are -3.333e-2, -0.1, -2.0, +0.1 and +1.0 for alpha, beta, gamma, delta and zeta respectively.
- **Constraints**: There are no constraints on the parameters for MAPEC, allowing for a fully adaptive response.
- **Stability**: A small constant `eps` (1.0e-3) is subtracted from denominator to prevent division by zero.

These functions aim to provide enhanced flexibility and adaptability for neural networks, particularly beneficial for complex pattern recognition tasks.

## Evaluation
To evaluate a model with a specific activation function on CIFAR-100 and plot _training loss*_, use:
```bash
python scripts/eval_cifar100.py --activation APEC --plot-loss
```

_* Plotting training loss requires `self-projection` package to be installed._

## Results
Evaluation results on CIFAR-100:

| Activation | Average Loss | Accuracy |
| ---------- | ------------ | -------- |
| MAPEC 16e* | 2.2004       | 43%      |
| APEC       | 2.2235       | 43%      |
| MAPEC 20e* | 2.2456       | 43%      |
| Mish       | 2.2704       | 43%      |
| SELU       | 2.2674       | 42%      |
| PReLU      | 2.2759       | 42%      |
| ReLU       | 2.3933       | 39%      |

_* Results provided for training with MAPEC activation for 20 and 16 epochs respectively._


**APEC** leads to the best performance, closely followed by Mish and SELU.

**MAPEC** leads to the faster convergence with performance closely followed by APEC.

You could look at training loss plots [here](doc/plots.md).

## Contributing
Contributions and suggestions are welcome! Feel free to fork the repository, open issues, and submit pull requests.

## License

`APEC` is released under the MIT License. See the `LICENSE` file for more details.
