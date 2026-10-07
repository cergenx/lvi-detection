"""
Post-processing on teh detection output of the SVM or CNN model
"""
import numpy as np
from scipy.signal import convolve


class PostProc():
    """Additional post-processing of the CNN output."""
    def __init__(self, fs=8, win_len=4, min_event_len=None):
        self.min_event_len = int(min_event_len * fs) if min_event_len is not None else None
        self.win = np.ones(int(fs * win_len)) / int(fs * win_len) if win_len is not None else None

    def post_processing(self, y_prob, y_pred=None):
        if self.win is not None:
            y_prob_bounded = True if np.all((y_prob >= 0) & (y_prob <= 1)) else False
            # Keep the input length even when the smoothing window is longer.
            y_prob = convolve(y_prob, self.win, mode='same', method='direct')
            if y_prob_bounded:
                y_prob = np.clip(y_prob, 0, 1)
            y_pred = np.array(y_prob > 0.5, dtype=np.int32)

        if self.min_event_len is not None and y_pred is not None:
            # Repeat filling and deleting operations until mask stops changing.
            previous_y_pred = None
            while np.array_equal(y_pred, previous_y_pred) is False:
                previous_y_pred = y_pred.copy()
                y_pred = self._fill_or_delete_short_segments(y_pred, mode='fill')
                y_pred = self._fill_or_delete_short_segments(y_pred, mode='delete')
        return y_prob, y_pred

    def _fill_or_delete_short_segments(self, mask, mode):
        """
        Fills or deletes short segments in a binary mask based on the specified mode.

        Parameters:
        mask (numpy.ndarray): The binary mask to process.
        mode (str): 'fill' to fill segments of zeros, 'delete' to delete segments of ones.

        Returns:
        numpy.ndarray: The processed binary mask.
        """
        assert mode in ['fill', 'delete'], "'mode' should be either 'fill' or 'delete'"
        l_mask = len(mask)

        def check_start_end_indices(istart, iend, length):
            return max(istart, 0), min(iend, length)

        find_zeros = (mode == 'fill')
        lengths, start_indices, _ = self._length_continuous_binary_values(mask, find_zeros=find_zeros)
        segments = np.where(lengths < self.min_event_len)[0]
        new_value = 0 if mode == 'delete' else 1  # Determine fill value based on mode
        for p in segments:
            istart, iend = check_start_end_indices(start_indices[p], start_indices[p] + lengths[p], l_mask)
            # ignore start/end of mask
            if istart > 0 and iend < l_mask:
                mask[istart:iend] = new_value
        return mask

    def _length_continuous_binary_values(self, binary_sequence, find_zeros=True):
        """
        Find the lengths of continuous runs of a certain binary value in a sequence.

        Parameters:
        binary_sequence (numpy.ndarray): The binary sequence to process.
        find_zeros (bool, optional): If True, find runs of zeros. Otherwise, find runs of ones. Defaults to True.

        Returns:
        tuple: The lengths, start indices and end indices of the runs.
        """
        binary_sequence = binary_sequence.astype(int)

        # Invert the sequence if we're looking for ones.
        binary_sequence = 1 - binary_sequence if not find_zeros else binary_sequence

        if all(binary_sequence == binary_sequence[0]):
            if binary_sequence[0] == 1:
                return np.array([]), np.array([]), np.array([])
            else:
                lens = np.array([len(binary_sequence)])
                start_indices = np.array([0])
                end_indices = lens
                return lens, start_indices, end_indices

        edge_indices = np.diff(np.hstack((0, binary_sequence == 0, 0)))
        start_indices = np.where(edge_indices == 1)[0]
        end_indices = np.where(edge_indices == -1)[0]
        lengths = end_indices - start_indices

        return lengths, start_indices, end_indices
