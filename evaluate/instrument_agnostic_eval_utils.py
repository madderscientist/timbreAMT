"""
用于音色无关转录的评估
包括帧级别和音符级别的评估，其中音符级别只关注onset时间和pitch偏移
"""
import torch
import torchaudio
import os
import numpy as np
import sys
sys.path.append('..')


### 预处理和保存结果，只保留onset和note和midi三项

def amt_one(model: torch.nn.Module, file: str):
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.unsqueeze(0)
    onset, note = model(waveform)
    onset = onset.cpu().numpy()[0]
    note = note.cpu().numpy()[0]
    return onset / onset.max(), note / note.max()

def amt_piece(model: torch.nn.Module, piece_folder: str):
    filename = os.listdir(piece_folder)[0]
    path = os.path.join(piece_folder, filename)[:-3]    # 去掉后缀
    onset, note = amt_one(model, path + "wav")
    midi = np.load(path+"npy")
    # 补时间长度
    freqs, times = note.shape
    if midi.shape[1] < times:
        padding = np.zeros((freqs, times - midi.shape[1]))
        midi = np.concatenate((midi, padding), axis=1)
    elif midi.shape[1] > times:
        midi = midi[:, :times]
    return onset, note, midi

def amt_dataset(model: torch.nn.Module, dataset_folder: str, output_folder: str = './'):
    folder_name = os.path.basename(dataset_folder)
    print(f"processing {folder_name}")
    output_folder_name = folder_name.split("_")[0] + "_eval"
    output_path = os.path.join(output_folder, output_folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for piece_folder in os.listdir(dataset_folder):
        if os.path.isdir(os.path.join(dataset_folder, piece_folder)):
            result = amt_piece(model, os.path.join(dataset_folder, piece_folder))
            np.save(os.path.join(output_path, piece_folder+".npy"), np.stack(result, axis=0))
            print(f"\tFinish {piece_folder}")


### 计算帧级的评价指标
from utils.midiarray import freq_map, roll2evalarray, array2notes
from utils.postprocess import min_len_
import mir_eval

_freqmap_offset = 24
_freqmap = freq_map((_freqmap_offset, _freqmap_offset + 83), 440)

def frame_eval(note: np.ndarray, midi: np.ndarray, threshold = 0.5, s_per_frame = 256 / 22050, freqmap = _freqmap):
    """
    对note进行阈值二值化、移除短音符、转换为mir_eval所需数
    计算帧级评价指标
    note: (freqs, times)
    midi: (freqs, times)
    """
    binary_note = (note > threshold).astype(int)    # 二值化    
    if binary_note.shape[1] != midi.shape[1]:
        if binary_note.shape[1] < midi.shape[1]:
            pad_width = midi.shape[1] - binary_note.shape[1]
            binary_note = np.pad(binary_note, ((0, 0), (0, pad_width)), mode='constant')
        else:
            pad_width = binary_note.shape[1] - midi.shape[1]
            midi = np.pad(midi, ((0, 0), (0, pad_width)), mode='constant')
    # 这个min_len_是原位操作，会修改输入
    est_pitch = roll2evalarray(min_len_(binary_note, 3), freqmap)
    ref_pitch = roll2evalarray(midi, freqmap)
    rst_time = s_per_frame * np.arange(len(est_pitch))
    ref_time = s_per_frame * np.arange(len(ref_pitch))
    result = mir_eval.multipitch.evaluate(ref_time, ref_pitch, rst_time, est_pitch)
    return result   # https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/multipitch.py 看源码


def evaluate_frame_dataset(npy_pathes, threshold = 0.5, log = True):
    """
    对npy_pathes中的所有npy文件用同一个阈值进行评估
    dataset_folder: folder containing npy files, each file is a result of amt_piece, shape (3, freqs, times): onset, note, midi
    """
    accs = []
    ps = []
    rs = []
    f1s = []
    for npy_file in npy_pathes:
        result = np.load(npy_file)
        if npy_file.endswith('.npz'):
            note = result['frame']
            midi = result['midi']
        else:
            note = result[1]
            midi = result[2]
        # 如果midi是三维的，取最大值
        if midi.ndim == 3:
            midi = midi.max(axis=0)
        evaluation = frame_eval(note, midi, threshold)
        acc = evaluation['Accuracy']
        p = evaluation['Precision']
        r = evaluation['Recall']
        accs.append(acc)
        ps.append(p)
        rs.append(r)
        f1s.append(2*p*r/(p+r) if p+r > 0 else 0)
    ACC = np.mean(accs)
    P = np.mean(ps)
    R = np.mean(rs)
    F1 = np.mean(f1s)
    if log:
        # | Acc | P | R | F1 |
        print(f"| {threshold:.5f} | {ACC:.5f} | {P:.5f} | {R:.5f} | {F1:.5f} |")
    return ACC, P, R, F1


def find_best_threshold(npy_pathes, origin_range = (0.1, 0.9), step_num = 10, generation = 4, log = True)->tuple:
    if log:
        print("| threshold | Acc | P | R | F1 |")
        print("| --------- | --- |---|---|----|")
    
    start = origin_range[0]
    end = origin_range[1]
    step = (end - start) / step_num
    
    best_thre = -1
    max_acc, max_p, max_r, max_f1 = -1, -1, -1, -1

    for g in range(generation):
        best_thre_idx = -1
        lastF1 = -1
        thresholds = np.r_[start:end:step]
        for idx, thre in enumerate(thresholds):
            ACC, P, R, F1 = evaluate_frame_dataset(npy_pathes, thre, log)
            if F1 > max_f1:
                max_acc, max_p, max_r, max_f1 = ACC, P, R, F1
                best_thre_idx = idx
                best_thre = thre
            if F1 < lastF1: # 假设F1是一个凹函数，只要开始下降就可以停止了
                break
            lastF1 = F1

        if log and g < generation - 1:
            print(f"| Best threshold | {best_thre:.5f} | ~ | ~ | F1: {max_f1:.5f} |")
        
        if best_thre_idx == -1:
            start = max(0, best_thre - step)
            end = best_thre + step
            step = (end - start) / (step_num + 1)
            start += step
        else:
            start = max(0, thresholds[best_thre_idx] - step)
            end = thresholds[best_thre_idx] + step
            step = (end - start) / (step_num + 1)
            start += step

    if log:
        print("| best | Acc | P | R | F1 |")
        print("| ---- | --- |---|---|----|")
        print(f"| {best_thre:.5f} | {max_acc:.5f} | {max_p:.5f} | {max_r:.5f} | {max_f1:.5f} |")
    return best_thre, max_acc, max_p, max_r, max_f1

### 计算音符级的评价指标
from utils.postprocess import output_to_notes_polyphonic

def note_eval(frame: np.ndarray, onset: np.ndarray, midi: np.ndarray, frame_thresh = 0.5, onset_thresh = 0.4, s_per_frame = 256 / 22050, freqmap = _freqmap):
    """
    计算音符级评价指标
    note: (freqs, times)
    onset: (freqs, times)
    midi: (freqs, times)
    frame_thresh: 阈值二值化frame
    onset_thresh: 阈值二值化onset
    """
    est_notes = output_to_notes_polyphonic(
        frame, onset, frame_thresh, onset_thresh,
        min_note_len=7.5,
        infer_onsets=True,
        melodia_trick=True,
        neighbor_trick=False,
        energy_tol = 11,
        midi_offset = 0
    )
    est_notes.sort(key=lambda x: x[0])  # 按onset排序

    actual_notes = array2notes(midi)
    actual_notes = [element for track in actual_notes for element in track]   # 合并所有音轨
    actual_notes.sort(key=lambda x: x[0])  # 按onset排序

    # 转换为mir_eval所需格式
    est_intervals = np.array([[onset * s_per_frame, offset * s_per_frame] for onset, offset, _, _ in est_notes])
    est_pitches = np.array([freqmap[note] for _, _, note, _ in est_notes])
    ref_intervals = np.array([[onset * s_per_frame, offset * s_per_frame] for onset, offset, _ in actual_notes])
    ref_pitches = np.array([freqmap[note] for _, _, note in actual_notes])
    result = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,
        # 不关心offset
        offset_ratio=None # type: ignore
    )   # precision, recall, f_measure, avg_overlap_ratio
    return result

def evaluate_note_dataset(npy_pathes, frame_thresh = 0.5, onset_thresh = 0.4, log = True):
    ps = []
    rs = []
    fs = []
    overlaps = []

    for npy_file in npy_pathes:
        result = np.load(npy_file)  # onset, note, midi
        if npy_file.endswith('.npz'):
            frame = result['frame']
            onset = result['onset']
            midi = result['midi']
        else:
            frame = result[1]
            onset = result[0]
            midi = result[2]
        # 如果midi是三维的，取最大值
        if midi.ndim == 3:
            midi = midi.max(axis=0)
        p, r, f, avg_overlap_ratio = note_eval(frame, onset, midi, frame_thresh=frame_thresh, onset_thresh=onset_thresh)
        ps.append(p)
        rs.append(r)
        fs.append(f)
        overlaps.append(avg_overlap_ratio)
    
    P = np.mean(ps)
    R = np.mean(rs)
    F = np.mean(fs)
    OVERLAP = np.mean(overlaps)

    if log:
        print(f"| {onset_thresh:.5f} | {P:.5f} | {R:.5f} | {F:.5f} | {OVERLAP:.5f} |")

    return P, R, F, OVERLAP


def find_best_onset_threshold(npy_pathes, frame_thresh = 0.5, origin_range = (0.1, 0.9), step_num = 10, generation = 4, log = True)->tuple:
    if log:
        print("| onset threshold | P | R | F | overlap |")
        print("| --------------- |---|---|---| ------- |")
    
    start = origin_range[0]
    end = origin_range[1]
    step = (end - start) / step_num
    
    best_thre = -1
    max_p, max_r, max_f, max_overlap = -1, -1, -1, -1

    for g in range(generation):
        best_thre_idx = -1
        lastF = -1
        thresholds = np.r_[start:end:step]
        for idx, thre in enumerate(thresholds):
            P, R, F, OVERLAP = evaluate_note_dataset(npy_pathes, frame_thresh=frame_thresh, onset_thresh=thre, log=log)
            if F > max_f:
                max_p, max_r, max_f, max_overlap = P, R, F, OVERLAP
                best_thre_idx = idx
                best_thre = thre
            if F < lastF: # 假设F是一个凹函数，只要开始下降就可以停止了
                break
            lastF = F

        if log and g < generation - 1:
            print(f"| Best onset threshold | {best_thre:.5f} | ~ | F: {max_f:.5f} | ~ |")
        
        if best_thre_idx == -1:
            start = max(0, best_thre - step)
            end = best_thre + step
            step = (end - start) / (step_num + 1)
            start += step
        else:
            start = max(0, thresholds[best_thre_idx] - step)
            end = thresholds[best_thre_idx] + step
            step = (end - start) / (step_num + 1)
            start += step

    if log:
        print("| best | P | R | F | overlap |")
        print("| ---- |---|---|---| ------- |")
        print(f"| {best_thre:.5f} | {max_p:.5f} | {max_r:.5f} | {max_f:.5f} | {max_overlap:.5f} |")
    return best_thre, max_p, max_r, max_f, max_overlap