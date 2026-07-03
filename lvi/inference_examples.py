"""
Overview of how to use SVM and ConvNeXt models for inference on a test signal.
"""
import argparse
from time import perf_counter

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import torch

from lvi.cnx import CNXNano
from lvi.svm_feature_set import gen_feature_set
from lvi.post_processing import PostProc


def create_svm_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearSVC(dual=False))
    ])
    return pipeline

def _gen_test_signal(fs=64, duration=60*60):
    n_samples = fs * duration
    x = np.random.randn(n_samples)

    print(
        f"1) Generating a test signal containing {n_samples} samples "
        f"({duration} seconds at {fs} Hz).",
        flush=True,
    )
    return x, fs

def _segment_signal(x, fs):
    """Divide the signal into overlapping 2-second segments shifted by one sample."""
    segment_len = 2 * fs
    x_segments = np.lib.stride_tricks.sliding_window_view(x, segment_len)
    x_segments = torch.as_tensor(x_segments.copy(), dtype=torch.float32)
    return x_segments.unsqueeze(1).unsqueeze(-1)  # (batch, 1, 128, 1)



def example_svm():
    """
    Demonstrate a simple proof-of-principle SVM inference workflow.
    """
    print("\n--- SVM Inference Example ---", flush=True)
    x, fs = _gen_test_signal()

    print("2) Train an SVM model.", flush=True)
    start_time = perf_counter()
    model = create_svm_model()
    model.fit(np.random.randn(100, 8), np.random.randint(0, 2, size=100))

    print("3) Generate test feature test.", flush=True)
    feats = gen_feature_set(x, fs)
    feats[np.isnan(feats)] = 0

    print("4) Predict with SVM model.", flush=True)
    y_prob = model.decision_function(feats.T)
    # sigmoid function:
    y_prob = 1 / (1 + np.exp(-y_prob))

    print(
        "5) Post-processing the model output with a 2.25-second rectangular "
        "smoothing window and a 1-second minimum event length.",
        flush=True,
    )
    post_proc = PostProc(fs=fs, win_len=2.25, min_event_len=1)
    y_prob, y_pred =  post_proc.post_processing(y_prob)

    elapsed_time = perf_counter() - start_time
    print(
        f"5) Complete in {elapsed_time:.2f} seconds. The probability array "
        f"has shape {y_prob.shape}, and the binary prediction array has "
        f"shape {y_pred.shape}.",
        flush=True,
    )



def example_cnx():
    """
    Demonstrate a simple proof-of-principle ConvNeXt inference workflow.
    """
    print("\n--- ConvNeXt Inference Example ---", flush=True)
    x, fs = _gen_test_signal()
    x_segments = _segment_signal(x, fs)

    print("2) Loading the ConvNeXt Nano model with random weights.")
    start_time = perf_counter()
    model = CNXNano()
    model.eval()

    print("3) Run inference on the test signal.", flush=True)
    with torch.inference_mode():
        logits = model(x_segments)
        y_prob = model.activate_logits(logits).cpu().numpy()

    y_pred = (y_prob > 0.5).astype(np.int32)

    print(
        "4) Post-processing the model output with a 1-second rectangular "
        "smoothing window and a 1-second minimum event length.",
        flush=True,
    )
    post_proc = PostProc(fs=fs, win_len=1, min_event_len=1)
    y_prob, y_pred = post_proc.post_processing(y_prob, y_pred)

    elapsed_time = perf_counter() - start_time
    print(
        f"5) Complete in {elapsed_time:.2f} seconds. The probability array "
        f"has shape {y_prob.shape}, and the binary prediction array has "
        f"shape {y_pred.shape}.",
        flush=True,
    )

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference example for SVM and ConvNeXt models.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=("svm", "cnx", "both"),
        default="both",
        help="either 'svm', 'cnx', or 'both' (default: both)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.model in ("svm", "both"):
        example_svm()
    if args.model in ("cnx", "both"):
        example_cnx()
