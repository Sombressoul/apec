# APEC and MAPEC Activation Functions for Neural Networks

- [APEC and MAPEC Activation Functions for Neural Networks](#apec-and-mapec-activation-functions-for-neural-networks)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Activation Functions](#activation-functions)
  - [Mathematical Formulation](#mathematical-formulation)
    - [APEC (Asymmetric Parametric Exponential Curvature)](#apec-asymmetric-parametric-exponential-curvature)
    - [MAPEC (Multiplicative Asymmetric Parametric Exponential Curvature)](#mapec-multiplicative-asymmetric-parametric-exponential-curvature)
  - [Usage](#usage)
  - [Results](#results)
  - [CIFAR-100 Evaluation Script](#cifar-100-evaluation-script)
  - [Contributing](#contributing)
  - [License](#license)
  - [Training Loss Plots](#training-loss-plots)


## Overview
This repository introduces two novel activation functions, APEC (Asymmetric Parametric Exponential Curvature) and its variant MAPEC (Multiplicative APEC), designed for deep learning models to capture complex patterns with improved performance. Functions have been tested on the CIFAR-100 dataset (results included) and on some of my experimental models (results not included).

## Installation
```bash
pip install apec-afn
```

## Activation Functions
- **APEC**: Offers a balance between flexibility and performance, as demonstrated by the improvement over traditional functions on CIFAR-100.
- **MAPEC**: An extension of APEC with an additional multiplicative term, allowing for an even richer model expressiveness and an observed faster convergence (up to 15%).

## Mathematical Formulation

### APEC (Asymmetric Parametric Exponential Curvature)
APEC is designed to introduce a non-linear response with an adjustable curvature, defined by:
$$f(x) = a + \frac{b - x}{(g - \exp(-x)) + \epsilon}$$

- **Initialization**: Parameters `a` and `b` are initialized with a normal distribution of zero mean and a standard deviation of 0.35. Parameter `g` is initialized with a mean of -1.375 and a standard deviation of 0.35.
- **Constraints**: The default constraints for `a`, `b`, and `g` are [-2.0, +2.0], [-2.5, +2.5], and [-2.5, -0.25], respectively.
- **Stability**: A small constant `eps` (1.0e-5) is added to prevent division by zero.

### MAPEC (Multiplicative Asymmetric Parametric Exponential Curvature)
MAPEC extends APEC by adding a multiplicative term, enhancing its flexibility:
$$f(x) = a + \frac{b - x}{g - \exp(-x)} + (x \cdot d)$$

- **Initialization**: Parameters `a`, `b`, and `d` are initialized to 0.0, and `g` is initialized to -1.0.
- **Constraints**: There are no constraints on the parameters for MAPEC, allowing for a fully adaptive response.

These functions aim to provide enhanced flexibility and adaptability for neural networks, particularly beneficial for complex pattern recognition tasks.

## Usage
To evaluate a model with a specific activation function on CIFAR-100 and plot _training loss*_, use:
```bash
python scripts/eval_cifar100.py --activation APEC --plot-loss
```

_* Plotting training loss requires `self-projection` package to be installed._

## Results
Evaluation results on CIFAR-100:

| Activation | Average Loss | Accuracy |
|------------|--------------|----------|
| APEC       | 2.2235       | 43%      |
| *MAPEC 20e | 2.3301       | 42%      |
| *MAPEC 15e | 2.2509       | 42%      |
| Mish       | 2.2704       | 43%      |
| SELU       | 2.2674       | 42%      |
| PReLU      | 2.2759       | 42%      |
| ReLU       | 2.3933       | 39%      |

APEC leads to the best performance, closely followed by Mish and SELU.
MAPEC leads to the faster convergence with performance closely to APEC.

_* Results provided for training with MAPEC activation for 20 and 15 epochs respectively._

## CIFAR-100 Evaluation Script
Included in this repository is an evaluation script for the CIFAR-100 dataset.

Use the following command to see available options:
```bash
python scripts/eval_cifar100.py --help
```

## Contributing
Contributions and suggestions are welcome! Feel free to fork the repository, open issues, and submit pull requests.

## License

`APEC` is released under the MIT License. See the `LICENSE` file for more details.

## Training Loss Plots
* APEC:

![APEC](doc/CIFAR100_APEC.png)

* MAPEC:

![MAPEC](doc/CIFAR100_MAPEC.png)

* Mish:

![Mish](doc/CIFAR100_Mish.png)

* SELU:

![SELU](doc/CIFAR100_SELU.png)

* PReLU:

![PReLU](doc/CIFAR100_PReLU.png)

* ReLU:

![ReLU](doc/CIFAR100_ReLU.png)

