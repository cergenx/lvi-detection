# LVI paper code

```text
NAME of paper
```

Reference implementation of the quantitative EEG features and machine-learning
methods accompanying the associated scientific paper. The repository focuses on
the signal-processing and model workflows used in the study: quantitative
features, an eight-feature linear support vector machine (SVM), and a compact
one-dimensional ConvNeXt model.

> [!NOTE]
> This is companion research code rather than a standalone analysis package.
> The repository includes an example EEG excerpt and the required filter
> coefficients, but does not include the study dataset or fitted model files.

## Contents

- [Installation](#installation)
- [1. Quantitative features](#1-quantitative-features)
- [2. SVM features and model](#2-svm-features-and-model)
- [3. ConvNeXt Nano](#3-convnext-nano)
- [Post-processing](#post-processing)
- [Repository structure](#repository-structure)
- [Citation and licence](#citation-and-licence)

## Installation

The project requires Python 3.14 or later. From the repository root, create the
environment and install the locked dependencies with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run the examples below from the repository root using `uv run`. The first call
to the SVM feature extractor may take slightly longer while Numba compiles the
Higuchi fractal-dimension routine.

## 1. Quantitative features

[`lvi/quant_features.py`](lvi/quant_features.py) implements four sample-aligned
features for a one-dimensional EEG signal:

| Feature name   | Description                                                                                           |
|----------------|-------------------------------------------------------------------------------------------------------|
| `envelope`     | Hilbert envelope followed by a one-second median filter                                               |
| `line_length`  | Line length in a one-second sliding window                                                            |
| `edo`          | Envelope Derivative Operator (EDO), followed by a one-second median filter                            |
| `peak_to_peak` | Proportion of samples exceeding ±12.5 µV in a one-second sliding window, followed by median filtering |

All four outputs have the same length as the input signal and use the sign
convention expected by the inter-burst interval (IBI) analysis.

### Run the quantitative-feature example

```bash
uv run python -m lvi.plot_examples quant_features
```

This loads the bundled 64-Hz EEG excerpt, calculates all four quantitative
features, and opens a figure containing the EEG and aligned feature traces.
Close the figure window to return to the shell.

## 2. SVM features and model

The SVM workflow comprises feature generation in
[`lvi/svm_feature_set.py`](lvi/svm_feature_set.py) and model training/inference
in [`lvi/svm_train_test.py`](lvi/svm_train_test.py).

### Generate the eight-feature set

`IBIParams` contains the analysis settings used by `gen_feature_set`, including
frequency bands, epoch lengths, overlap, filtering, and log transforms. With an
input of `N` samples, the returned feature matrix has shape `(8, N)`.

| Feature name            | Feature                      | Frequency range                |
|-------------------------|------------------------------|--------------------------------|
| `envelope_#1`           | Hilbert envelope             | 0.5–3 Hz                       |
| `fd_higuchi`            | Higuchi fractal dimension    | 0.5–30 Hz                      |
| `edo`                   | Envelope Derivative Operator | 0.5–10 Hz                      |
| `if_#4`                 | Instantaneous frequency      | 15–30 Hz                       |
| `psd_r2_#1`             | Goodness of log–log PSD fit  | 0.5–3 Hz                       |
| `envelope_#4`           | Hilbert envelope             | 15–30 Hz                       |
| `envelope_#3`           | Hilbert envelope             | 8–15 Hz                        |
| `rel_spectral_power_#4` | Relative spectral power      | 15–30 Hz relative to 0.5–30 Hz |

```python
import numpy as np

from lvi.svm_feature_set import IBIParams, gen_feature_set

fs = 64
x = np.load("data/eeg_example_fs64.npy")
params = IBIParams()

x_features = gen_feature_set(x.copy(), fs, params)
print(params.feature_names)
print(x_features.shape)  # (8, number_of_signal_samples)
```

Use `x.copy()` when the original array must remain unchanged: the feature
extractor temporarily replaces missing input samples while calculating the
features.

To plot the eight SVM features:

```bash
uv run python -m lvi.plot_examples svm_features
```

Both the quantitative-feature and SVM-feature figures can be opened in
sequence with:

```bash
uv run python -m lvi.plot_examples all
```

Running the module without an argument also defaults to `all`.

### Run the SVM demonstration

See the example in `example_train_and_test_svm_with_random_data()` in [`lvi/svm_train_test.py`](lvi/svm_train_test.py).
This example generates test signals and labels, extracts the feature
set, trains a linear SVM, writes it to a temporary file, reloads it, and
runs inference.

## 3. ConvNeXt Nano

[`lvi/cnx.py`](lvi/cnx.py) defines the ConvNeXt model as a PyTorch Lightning module. `CNXNano`
accepts single-channel segments of 2-seconds sampled at 64 Hz. A batch has the shape `(batch, 1,
128, 1)`, and the model produces one value (probability) per segment.

See `example_cnx_nano()` in [`lvi/cnx.py`](lvi/cnx.py). This function generates a 1-hour test
signal, constructs two-second windows shifted by one sample, performs inference, and applies
one-second smoothing and minimum-event (1-second) post-processing.

```bash
uv run python -m lvi.cnx
```

The example instantiates `CNXNano` with randomly initialised weights so that
the complete tensor and post-processing path can be exercised without a
checkpoint. To use fitted weights from a compatible Lightning checkpoint,
replace the model construction in `example_cnx_nano()` with:

```python
model = CNXNano.load_from_checkpoint("path/to/model.ckpt")
model.eval()
```


## Repository structure

```text
.
├── data/
│   ├── eeg_example_fs64.npy    # example 64-Hz EEG excerpt
│   └── ellip_filt_coeffs.npz   # filter coefficients used for the SVM features
├── lvi/
│   ├── quant_features.py       # four quantitative amplitude features used to detect IBI
│   ├── svm_feature_set.py      # feature set for the SVM model
│   ├── svm_train_test.py       # linear SVM training and inference
│   ├── cnx.py                  # ConvNeXt Nano model and inference example
│   ├── post_processing.py      # shared output post-processing
│   ├── plot_examples.py        # feature visualisations
│   └── utils.py                # filtering and signal-processing utilities
├── pyproject.toml
└── uv.lock
```

## Citation and licence

If this code contributes to published work, please cite the accompanying
scientific paper. Citation details can be added here when the paper record is
available.

The code is released under the [MIT License](LICENSE).
