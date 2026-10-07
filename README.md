# Detecting Low-Voltage Intervals (LVI) in Neonatal EEG

Implementation of the quantitative EEG features and machine-learning methods used in:

> JM O'Toole, SR Mathieson, A Luca, S Ventura, S Griffin, GB Boylan, and R Hogan.
> *Expert-level detection of low-voltage intervals in EEG as a robust biomarker of neonatal
> encephalopathy.* Under review, 2026.

The repository includes the signal-processing and model workflows used in the study: quantitative
features, an eight-feature linear support vector machine (SVM), and a compact one-dimensional
ConvNeXt model.

> [!NOTE]
> This is companion research code rather than a standalone analysis package.

## Contents

- [Installation](#installation)
- [Quantitative features](#quantitative-features)
- [SVM features and model](#svm-features-and-model)
- [ConvNeXt Nano](#convnext-nano)
- [Repository structure](#repository-structure)
- [Citation and licence](#citation-and-licence)

## Installation

The project requires Python 3.14 or later. From the repository root, create the environment and
install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run the examples below from the repository root using `uv run`. The first call to the SVM feature
extractor may take slightly longer while Numba compiles the Higuchi fractal-dimension routine.

## Quantitative features

[`lvi/quant_features.py`](lvi/quant_features.py) implements the four measures used to detect LVIs:

| Feature        | Description                                              | Post-processing   |
|----------------|----------------------------------------------------------|-------------------|
| `envelope`     | Hilbert envelope                                         | 2 s median filter |
| `line_length`  | Absolute value of first-order difference                  | 1 s sum           |
| `edo`          | Envelope Derivative Operator (EDO)                        | 2 s median filter |
| `peak_to_peak` | Proportion of samples exceeding ±12.5 µV over a 1 s window | 2 s median filter |

Each feature is negated so a larger score indicates lower-voltage activity. Input is a 1D signal in µV.

Plot the four quantitative features:

```bash
uv run python -m lvi.plot_examples quant_features
```

This loads the bundled 64-Hz EEG excerpt, calculates all four quantitative features, and opens a
figure containing the EEG and aligned feature traces. Close the figure window to return to the
shell.

## SVM features and model

The SVM workflow consists of feature generation in
[`lvi/svm_feature_set.py`](lvi/svm_feature_set.py) and example training/inference in
[`lvi/inference_examples.py`](lvi/inference_examples.py).

### Generate the eight-feature set

`IBIParams` contains the analysis settings used by `gen_feature_set`, including frequency bands,
epoch lengths, overlap, filtering, and log transforms. With an input of `N` samples, the returned
feature matrix has shape `(8, N)`.

| Feature name            | Signal summary               | Frequency range              |
|-------------------------|------------------------------|------------------------------|
| `envelope_#1`           | Hilbert envelope             | 0.5–3 Hz                     |
| `fd_higuchi`            | Higuchi fractal dimension    | 0.5–30 Hz                    |
| `edo`                   | Envelope Derivative Operator | 0.5–10 Hz                    |
| `if_#4`                 | Instantaneous frequency      | 15–30 Hz                     |
| `psd_r2_#1`             | Goodness of log–log PSD fit   | 0.5–3 Hz                     |
| `envelope_#4`           | Hilbert envelope             | 15–30 Hz                     |
| `envelope_#3`           | Hilbert envelope             | 8–15 Hz                      |
| `rel_spectral_power_#4` | Relative spectral power      | 15–30 Hz relative to 0.5–30 Hz |

To plot the eight SVM features:

```bash
uv run python -m lvi.plot_examples svm_features
```

### SVM inference

See the function `example_svm()` in [`lvi/inference_examples.py`](lvi/inference_examples.py). This
example fits a linear SVM on random features and labels, generates a synthetic test signal, extracts
its feature set, and runs inference with post-processing.

```bash
uv run python -m lvi.inference_examples svm
```

## ConvNeXt Nano

[`lvi/cnx.py`](lvi/cnx.py) defines the ConvNeXt model as a PyTorch Lightning module. `CNXNano()`
accepts single-channel 2 s segments sampled at 64 Hz. A batch has the shape `(n_batch, 1, 128, 1)`,
and the model produces one value (logit) per segment.

See `example_cnx()` in [`lvi/inference_examples.py`](lvi/inference_examples.py). This function
generates a 1-hour synthetic test signal, constructs 2 s windows shifted by one sample, performs
inference using randomly initialized weights, and applies 1 s smoothing and minimum-event (1 s)
post-processing.

```bash
uv run python -m lvi.inference_examples cnx
```

## Repository structure

```text
.
├── data/
│   ├── eeg_example_fs64.npy    # example 64-Hz EEG excerpt
│   └── ellip_filt_coeffs.npz   # filter coefficients used for the SVM features
├── lvi/
│   ├── quant_features.py       # quantitative features for inter-burst intervals (IBIs)
│   ├── svm_feature_set.py      # feature set for the SVM model
│   ├── inference_examples.py   # inference examples for SVM and ConvNeXt models
│   ├── cnx.py                  # ConvNeXt Nano model
│   ├── post_processing.py      # shared output post-processing
│   ├── plot_examples.py        # feature visualisations
│   └── utils.py                # filtering and signal-processing utilities
├── pyproject.toml
└── uv.lock
```

## Citation and licence

If this code contributes to published work, please cite the accompanying scientific paper.

The code is released under the [MIT License](LICENSE).
