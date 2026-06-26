"""

Train and test the SVM model using pre-generated features.

See function example_train_and_test_svm_with_random_data() below for an example workflow that trains
an SVM and runs inference using random data
"""
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from scipy.signal import resample_poly
from joblib import dump, load

from lvi.svm_feature_set import gen_feature_set, IBIParams
from lvi.post_processing import PostProc



class TrainSVM:
    """Train the SMV model using the pre-generated features."""
    def __init__(self):
        self.param_grid = {'model__C': [0.0001, 0.001, 0.01, 0.1, 1]}

    def _create_model(self, c_param=1):
        svm_params = {
            'penalty': 'l2',
            'C': c_param,
            'class_weight': None,
            'dual': False,
            'verbose': 0
        }
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearSVC(**svm_params))
        ])
        return pipeline

    def train_model(self, x_train, y_train, opt_params=False):
        """
        Train the SVM model.

        x_train: DataFrame containing training features. (columns: 'x_<feature1>', ... 'x_<featureN>')
        y_train: Series containing training labels (0 or 1). (column: 'y_train')
        opt_params: Boolean indicating whether to perform hyperparameter optimization using GridSearchCV.

        Returns:
        trained_model: The trained SVM model (Pipeline object).
        """
        pipeline = self._create_model()
        if opt_params:
            pipeline = GridSearchCV(pipeline, self.param_grid, cv=5, n_jobs=8, verbose=1)
        else:
            pipeline = self._create_model()

        pipeline.fit(x_train, y_train)

        if opt_params:
            best_params = pipeline.best_params_
            best_C = best_params['model__C']
            print(f'Best C parameter selected by nested cross-validation: {best_C}')
            trained_model = pipeline.best_estimator_
        else:
            trained_model = pipeline

        return trained_model


class TestSVM:
    """Inference run for trained SVM."""
    def __init__(self, model_filename=None, post_proc=True, fs_down=None):
        assert model_filename is not None, "Model filename must be provided"
        self.model = load(model_filename)

        self.ibi_params = IBIParams()
        self.fs = 64
        if fs_down is not None:
            assert self.fs / fs_down == int(self.fs / fs_down), "Downsampling factor must be an integer"
            self.downsample_factor = int(self.fs / fs_down)
        else:
            self.downsample_factor = None

        fs = self.fs if self.downsample_factor is None else fs_down
        self.post_proc = PostProc(fs=fs, win_len=2.25, min_event_len=1) if post_proc else None


    def test_model(self, x):
        x_feats = gen_feature_set(x, self.fs, self.ibi_params)
        x_feats[np.isnan(x_feats)] = 0
        if self.downsample_factor is not None:
            x_feats = np.asarray(
                [resample_poly(x_feats[n, :], 1, self.downsample_factor) for n in range(x_feats.shape[0])]
            )

        x_feat_df = pd.DataFrame(x_feats.T, columns=[f"x_{f}" for f in self.ibi_params.feature_names])
        x_feat_df = x_feat_df[self.model.feature_names_in_]

        y_pred = self.model.predict(x_feat_df)
        y_prob = self.model.decision_function(x_feat_df)
        # sigmoid function:
        y_prob = 1 / (1 + np.exp(-y_prob))

        if self.post_proc is not None:
            y_prob, y_pred =  self.post_proc.post_processing(y_prob, y_pred)

        return y_pred, y_prob





def example_train_and_test_svm_with_random_data():
    """Example workflow for training an SVM and running inference.

    This uses random data so the example can run without the original dataset.
    Replace ``x_train_raw``, ``y_train``, and ``x_test_raw`` with real data when
    using this code in practice.
    """

    rng = np.random.default_rng(0)
    fs = 64
    params = IBIParams()

    # In practice, replace this with a real training signal.
    x_train_raw = rng.normal(size=fs * 20)
    x_train_feats = gen_feature_set(x_train_raw, fs, params)
    x_train_feats[np.isnan(x_train_feats)] = 0
    x_train = pd.DataFrame(
        x_train_feats.T,
        columns=[f"x_{feature_name}" for feature_name in params.feature_names],
    )

    # In practice, replace this with real 0/1 labels aligned to x_train rows.
    y_train = rng.integers(0, 2, size=len(x_train))
    y_train[0] = 0
    y_train[1] = 1

    trained_model = TrainSVM().train_model(x_train, y_train)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_filename = Path(temp_dir) / "random_svm_model.joblib"
        dump(trained_model, model_filename)

        # In practice, replace this with a real test signal sampled at fs.
        x_test_raw = rng.normal(size=fs * 12)
        y_pred, y_prob = TestSVM(
            model_filename=model_filename,
            post_proc=False,
        ).test_model(x_test_raw)

    return y_pred, y_prob
