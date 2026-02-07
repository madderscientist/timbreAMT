from typing import List, Tuple, Union
import numpy as np
import scipy.signal

# 从basicpitch拿来的创建音符函数
def get_infered_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 3) -> np.ndarray:
    """Infer onsets from large changes in frame amplitudes.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        # 在时间维度上，将frames向后移动n帧
        frames_appended = np.concatenate([np.zeros((frames.shape[0], n)), frames], axis=1)
        # 计算差值=frames[t+n] - frames[t]: 后n帧强度减去本帧强度
        diffs.append(frames_appended[:, n:] - frames_appended[:, :-n])

    frame_diff = np.min(diffs, axis=0)  # 选取变化最小的作为最大值，比较保守
    frame_diff[frame_diff < 0] = 0      # 忽略前面比后面大的差值
    frame_diff[:, :n_diff] = 0          # 忽略前n_diff帧，因为是补零的
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)  # 缩放到和onsets相同的范围

    max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences
    return max_onsets_diff


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    frame_thresh: float,
    onset_thresh: float,
    min_note_len: Union[int, float] = 6,
    infer_onsets: bool = True,
    melodia_trick: bool = True,
    energy_tol: int = 12,
    midi_offset = 24,
    neighbor_trick = True
) -> List[Tuple[int, int, int, float]]:
    """Decode raw model output to polyphonic note events

    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        melodia_trick : Whether to use the melodia trick to better detect notes. 不依赖onset，根据frames中的极大值找额外的音符
        energy_tol: Drop notes below this energy.
        midi_offset: 从idx映射到midi音符编码的偏移
        neighbor_trick: 确认为音符后是否将相邻半音清空

    Returns:
        list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
        representing the note events, where amplitude is a number between 0 and 1
    """

    n_frames = frames.shape[1]

    # 根据frame在时间上的强度变化推断onset
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat.T >= onset_thresh) # 为了按照时间排序，所以转置（原本时间在第二维）
    onset_time_idx = onset_idx[0][::-1]  # 时间从后到前
    onset_freq_idx = onset_idx[1][::-1]  # 频率跟着反

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # 如果剩下的距离不够放一个最短的音符，就跳过
        if note_start_idx >= n_frames - min_note_len:
            continue

        i = note_start_idx + 1
        k = 0  # 连续k个小于frame_thresh的帧
        freqArray = remaining_energy[freq_idx]
        # 向后搜索，连续energy_tol帧小于frame_thresh（或者到达最后一帧），就认为这个音符结束。目的是将分散的frames合并
        while i < n_frames - 1 and k < energy_tol:
            if freqArray[i] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # 回到音符结尾

        # 跳过太短的音符
        if i - note_start_idx <= min_note_len:
            continue
        # 确定是音符了，把对应的强度置零
        freqArray[note_start_idx:i] = 0
        # 下面这段大概是认为半音不会同时出现。确实有一定合理性
        if neighbor_trick:
            if freq_idx < 83:
                remaining_energy[freq_idx + 1, note_start_idx:i] = 0
            if freq_idx > 0:
                remaining_energy[freq_idx - 1, note_start_idx:i] = 0

        # 取平均强度为其音符力度
        amplitude = np.mean(frames[freq_idx, note_start_idx:i])
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx + midi_offset,
                amplitude,  # 归一化
            )
        )

    if melodia_trick:   # 不依赖onset，根据frames中的极大值找额外的音符
        energy_shape = remaining_energy.shape

        while np.max(remaining_energy) > frame_thresh:
            freq_idx, i_mid = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            freqArray = remaining_energy[freq_idx]
            freqArray[i_mid] = 0

            # 向后搜索
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if freqArray[i] < frame_thresh:
                    k += 1
                else:
                    k = 0

                freqArray[i] = 0
                if freq_idx < 83:
                    remaining_energy[freq_idx + 1, i] = 0
                if freq_idx > 0:
                    remaining_energy[freq_idx - 1, i] = 0

                i += 1

            i_end = i - 1 - k  # go back to frame above threshold

            # 向前搜索
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if freqArray[i] < frame_thresh:
                    k += 1
                else:
                    k = 0

                freqArray[i] = 0
                if neighbor_trick:
                    if freq_idx < 83:
                        remaining_energy[freq_idx + 1, i] = 0
                    if freq_idx > 0:
                        remaining_energy[freq_idx - 1, i] = 0

                i -= 1

            i_start = i + 1 + k  # go back to frame above threshold
            # assert i_start >= 0, "{}".format(i_start)
            # assert i_end < n_frames
            i_start = max(0, i_start)
            i_end = min(n_frames, i_end)

            if i_end - i_start <= min_note_len:
                # note is too short, skip it
                continue

            # add the note
            amplitude = np.mean(frames[freq_idx, i_start:i_end])
            note_events.append(
                (
                    i_start,
                    i_end,
                    freq_idx + midi_offset,
                    amplitude,
                )
            )

    return note_events


def cluster_notes(
    frames: np.ndarray,
    onsets: np.ndarray,
    emb: np.ndarray,
    n_clusters: int,
    midi_offset = 24,
    **note_kwargs
)-> List[List[Tuple[int, int, int, float]]]:
    note_events = output_to_notes_polyphonic(
        frames,
        onsets,
        midi_offset = midi_offset,
        **note_kwargs
    )
    embeddings = []
    for start, end, f, amp in note_events:
        f = f - midi_offset
        _mask = frames[f, start:end]  # (end-start, )
        _emb = emb[:, f, start:end] # (18, end-start)
        # weight = np.ones_like(_mask)
        weight = _mask
        # weight = _mask * _mask
        # weight = np.sqrt(_mask)
        # weight = np.exp(_mask * _mask)
        weighted_emb = (_emb * weight).sum(axis=1)  # (18, )
        normalized_emb = weighted_emb / np.linalg.norm(weighted_emb)
        embeddings.append(normalized_emb)

    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels="cluster_qr")
    # ang_dist = 1 - cosine_similarity(np.array(embeddings))
    # sigma = np.percentile(ang_dist, 50)
    # affinity = np.exp(-ang_dist / sigma**2)
    # experiments show that, exp(cos_sim) works better than exp(-(1 - cos_sim) / sigma**2)
    affinity = np.exp(cosine_similarity(np.array(embeddings)))
    labels = spectral.fit_predict(affinity)

    clustered_notes = [[] for _ in range(n_clusters)]
    for label, note in zip(labels, note_events):
        clustered_notes[label].append(note)
    
    return clustered_notes

import gc
def cluster_frames(
    frames: np.ndarray, # (n_freqs, n_frames)
    emb: np.ndarray,    # (emb_dim, n_freqs, n_frames)
    n_clusters: int,
    frame_thresh: float,
) -> np.ndarray:    # (n_clusters, n_freqs, n_frames)
    mask = frames > frame_thresh  # (n_freqs, n_frames)
    embeddings = emb[:, mask].T  # (num_selected_frames, emb_dim)

    if embeddings.shape[0] > 22000:
        print(f"Warning: clustering {embeddings.shape[0]} frames, switch to KMeans.")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
    else:
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics.pairwise import cosine_similarity
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels="cluster_qr")
        affinity = np.exp(cosine_similarity(np.array(embeddings)))
        labels = spectral.fit_predict(affinity)
        del affinity

    gc.collect()
    result = np.zeros((n_clusters, *frames.shape), dtype=frames.dtype)
    freq_idx, time_idx = np.where(mask)
    # 将frames中被mask选中的元素按labels分类，填充到result
    result[labels, freq_idx, time_idx] = frames[freq_idx, time_idx]
    return result


def OTSU_threshold(image: np.ndarray, bins: int = 256) -> float:
    """
    Calculate the threshold value using the OTSU method.
    :param image: The image to calculate the threshold value for.
    :param bins: The number of bins to use for the histogram.
    :return: The threshold value.
    """
    hist, bin_edges = np.histogram(image.ravel(), bins=bins)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    index_of_max_val = np.argmax(inter_class_variance)
    return bin_mids[:-1][index_of_max_val]


def min_len_(arr: np.ndarray, min_len: int = 3) -> np.ndarray:
    if arr.ndim == 1:
        return min_len_1d_(arr, min_len)
    else:
        for index in np.ndindex(arr.shape[:-1]):
            arr[index] = min_len_1d_(arr[index], min_len)
        return arr


def min_len_1d_(arr: np.ndarray, min_len: int = 3) -> np.ndarray:
    """
    Remove sequences of 1s that are shorter than the specified minimum length.
    will modify the input array in place.
    """
    count = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            if count > 0:
                if count < min_len:
                    arr[i-count:i] = 0
                count = 0
        else:
            count += 1
    if count > 0 and count < min_len:
        arr[-count:] = 0
    return arr


def max_len_(arr: np.ndarray, max_len: int = 3, reset_len = float('inf')) -> np.ndarray:
    if arr.ndim == 1:
        return max_len_1d_(arr, max_len)
    else:
        for index in np.ndindex(arr.shape[:-1]):
            arr[index] = max_len_1d_(arr[index], max_len, reset_len)
        return arr


def max_len_1d_(arr: np.ndarray, max_len: int = 3, reset_len = float('inf')) -> np.ndarray:
    """
    Remove sequences of 1s that are longer than the specified maximum length.
    will modify the input array in place.
    """
    count = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            count = 0
        else:
            count += 1
            if count >= reset_len:
                count = 0
            elif count > max_len:
                arr[i] = 0
    return arr

def schmitt_binarization(image: np.ndarray, thres_min: float, thres_max: float) -> np.ndarray:
    """
    Apply Schmitt trigger binarization to the input image.
    :param image: The input image to binarize.
    :param thres_min: The minimum threshold value.
    :param thres_max: The maximum threshold value.
    :return: The binarized image.
    """
    if image.ndim == 1:
        return schmitt_binarization_1d(image, thres_min, thres_max)
    else:
        for index in np.ndindex(image.shape[:-1]):
            image[index] = schmitt_binarization_1d(image[index], thres_min, thres_max)
        return image

def schmitt_binarization_1d(arr: np.ndarray, thres_min: float, thres_max: float) -> np.ndarray:
    """
    Apply Schmitt trigger binarization to the input array.
    :param arr: The input array to binarize.
    :param thres_min: The minimum threshold value.
    :param thres_max: The maximum threshold value.
    :return: The binarized array.
    """
    output = np.zeros_like(arr)
    state = 0
    for i in range(len(arr)):
        if state == 0 and arr[i] > thres_max:
            state = 1
        elif state == 1 and arr[i] < thres_min:
            state = 0
        output[i] = state
    return output