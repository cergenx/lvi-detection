"""
Example plots for amplitude and SVM feature sets.
"""
import numpy as np
from matplotlib import pyplot as plt

try:
    from lvi import quant_features, svm_feature_set, utils
except ModuleNotFoundError:
    import quant_features
    import svm_feature_set
    import utils


FS = 64
EEG_COLOUR = '#064e3b'
THRESHOLD_COLOUR = '#7c2d12'
FEATURE_COLOURS_4 = ['#2e1065', '#5b21b6', '#9333ea', '#c084fc']
FEATURE_COLOURS_8 = [
    '#2e1065',
    '#4c1d95',
    '#5b21b6',
    '#6d28d9',
    '#7e22ce',
    '#9333ea',
    '#a855f7',
    '#c084fc',
]


def _load_example_eeg(fs=FS):
    eeg_filename = utils.DATA_DIR / f"eeg_example_fs{fs}.npy"
    assert eeg_filename.exists(), f"EEG file not found: {eeg_filename}"
    x = np.load(eeg_filename)
    t = np.arange(len(x)) / fs
    return x, t


def _style_axis(ax, show_xlabel=False, labelsize=9):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, labelsize=labelsize, colors='#4b5563')
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax.set_facecolor('white')
    if not show_xlabel:
        ax.tick_params(labelbottom=False)


def _add_eeg_axis(fig, gs, t, x, labelsize=9):
    ax_eeg = fig.add_subplot(gs[0, 0])
    ax_eeg.plot(t, x, color=EEG_COLOUR, linewidth=0.6)
    ax_eeg.axhline(12.5, color=THRESHOLD_COLOUR, linewidth=0.8, linestyle='--', alpha=0.85)
    ax_eeg.axhline(-12.5, color=THRESHOLD_COLOUR, linewidth=0.8, linestyle='--', alpha=0.85)

    range_x = t[-1] * 1.05
    ax_eeg.annotate(
        '',
        xy=(range_x, 12.5),
        xytext=(range_x, -12.5),
        arrowprops=dict(arrowstyle='<->', color=THRESHOLD_COLOUR, linewidth=1.0),
    )
    ax_eeg.text(
        range_x,
        0,
        '25 \u03bcV',
        color=THRESHOLD_COLOUR,
        fontsize=9,
        ha='right',
        va='center',
        backgroundcolor='white',
    )
    ax_eeg.set_yticks([-50, 0, 50])
    ax_eeg.set_ylabel('EEG (CZ-C3)', color=EEG_COLOUR, fontsize=10)
    _style_axis(ax_eeg, labelsize=labelsize)
    return ax_eeg


def _make_feature_figure(n_features, figsize, fignum=1):
    fig = plt.figure(fignum, clear=True)
    fig.set_size_inches(*figsize)
    gs = fig.add_gridspec(
        nrows=n_features + 1,
        ncols=1,
        height_ratios=[1.2, *([1] * n_features)],
        hspace=0.08,
    )
    return fig, gs


def _plot_feature_rows(fig, gs, ax_eeg, t, features, labels, colours, labelsize=9, plot_mask=None):
    if plot_mask is None:
        plot_mask = np.ones(len(t), dtype=bool)

    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[i + 1, 0], sharex=ax_eeg)
        colour = colours[i]
        ax.plot(t[plot_mask], features[i][plot_mask], color=colour, linewidth=1.0)
        ax.set_ylabel(label, color=colour, fontsize=labelsize)
        _style_axis(ax, show_xlabel=(i == len(labels) - 1), labelsize=labelsize)


def _finish_feature_figure(fig, left, bottom):
    fig.supxlabel('Time (s)', fontsize=11, y=0.03, color='#374151')
    fig.subplots_adjust(left=left, right=0.98, top=0.98, bottom=bottom)
    plt.show()


def plot_quant_feature_examples():
    """Plot examples of the amplitude features for IBI detection."""
    x, t = _load_example_eeg()
    feature_labels = {
        'envelope': 'Envelope',
        'line_length': 'Line Length',
        'edo': 'EDO',
        'peak_to_peak': 'Peak-to-Peak',
    }
    feature_types = list(feature_labels)
    features = [
        quant_features.calculate_feature(x, feature=feature_type, fs=FS)
        for feature_type in feature_types
    ]

    fig, gs = _make_feature_figure(len(feature_types), figsize=(12, 7), fignum=1)
    ax_eeg = _add_eeg_axis(fig, gs, t, x, labelsize=9)
    _plot_feature_rows(
        fig,
        gs,
        ax_eeg,
        t,
        features,
        [feature_labels[feature_type] for feature_type in feature_types],
        FEATURE_COLOURS_4,
        labelsize=9,
    )
    _finish_feature_figure(fig, left=0.11, bottom=0.09)


def plot_svm_feature_examples():
    """Plot examples of the SVM feature set for IBI detection."""
    x, t = _load_example_eeg()
    feature_plot_mask = (t >= 2) & (t <= (t[-1] - 2))
    feature_labels = {
        'envelope_#1': 'Envelope (FB1)',
        'fd_higuchi': 'FD',
        'edo': 'EDO',
        'if_#4': 'IF (FB4)',
        'psd_r2_#1': 'PSD fit (FB1)',
        'envelope_#4': 'Envelope (FB4)',
        'envelope_#3': 'Envelope (FB3)',
        'rel_spectral_power_#4': 'RSP (FB4)',
    }

    params = svm_feature_set.IBIParams()
    features = svm_feature_set.gen_feature_set(np.copy(x), FS, params=params)
    features_to_plot = np.copy(features)
    for i, feature_type in enumerate(params.feature_set_final):
        if feature_type in params.log_feats:
            features_to_plot[i, :] = np.exp(features_to_plot[i, :])

    fig, gs = _make_feature_figure(len(params.feature_names), figsize=(12, 10), fignum=2)
    ax_eeg = _add_eeg_axis(fig, gs, t, x, labelsize=8)
    _plot_feature_rows(
        fig,
        gs,
        ax_eeg,
        t,
        features_to_plot,
        [feature_labels[feature_name] for feature_name in params.feature_names],
        FEATURE_COLOURS_8,
        labelsize=8,
        plot_mask=feature_plot_mask,
    )
    _finish_feature_figure(fig, left=0.13, bottom=0.08)


if __name__ == "__main__":
    plot_quant_feature_examples()
    plot_svm_feature_examples()
