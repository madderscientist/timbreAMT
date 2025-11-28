"""
音色分离转录评估相关工具
"""
from itertools import combinations
import mir_eval
import numpy as np
import os
import torch
import torchaudio
import sys
sys.path.append('..')
from collections import defaultdict
from utils.midiarray import array2notes, midiarray_add
from functools import reduce

"""
命名：{name}_{a}&{b}&{c}.npz : 表示a,b,c三种乐器混合
每个npz包含emb, onset, frame, midi
emb: (D, F, T)  D维度已经归一化
onset: (F, T) 已经除以了max
frame: (F, T) 已经除以了max
midi: (num_instruments, F, T)
"""

def ifNmix(f, mix):
    """
    judge whether f is mixed with {mix} instruments
    根据命名规则判断
    """
    mix_info = f.split("_")[-1]
    mix_num = mix_info.split("&")
    return len(mix_num) == mix

def _merge_waves(wave_list):
    # 每个wave都是一维tensor，长度可能不一样，补0后相加
    # [1, len_i]
    # return [1, maxlen]
    max_length = max(wave.shape[1] for wave in wave_list)
    mix_wave = torch.zeros(1, max_length)
    for wave in wave_list:
        mix_wave[:, :wave.shape[1]] += wave
    return mix_wave

def _stack_midis(midi_list):
    # midi_list: list of (84, T_i)
    max_T = max(m.shape[1] for m in midi_list)
    padded_midi = [np.pad(m, ((0, 0), (0, max_T - m.shape[1]))) for m in midi_list]
    ensemble_midi = np.stack(padded_midi)   # (num_instruments, 84, maxT)
    return ensemble_midi


# 进行URMP的预处理
def urmp_folder_iterator(folder_path = "./URMP_processed", mix = 2):
    """
    mix: number of instruments to mix, -1 means all instruments
    yield: ensemble_name, ensemble_midi (mix, 84, T), ensemble_wave (num_samples,)
    """
    if mix < 2 and mix != -1:
        raise ValueError("mix should be at least 2 or -1 for all instruments")
    for piece_folder_name in os.listdir(folder_path):
        # 跳过文件
        piece_path = os.path.join(folder_path, piece_folder_name)
        if not os.path.isdir(piece_path):
            continue

        at_split = piece_folder_name.split('@')
        # 只看合奏
        if len(at_split) != 2 or at_split[1] != '0':
            continue

        # 选出乐器
        ensemble_name = at_split[0]
        instruments = ensemble_name.split('_')[2:]  # 前两个为id和name

        # 统计相同id的分组，从1开始
        id_groups = defaultdict(list)
        for idx, inst_id in enumerate(instruments, 1):
            id_groups[inst_id].append(idx)
        groups = list(id_groups.values())
        if len(groups) < 2:
            continue
        if mix != -1 and len(groups) != mix:
            continue

        piece_path = os.path.join(folder_path, ensemble_name + '@')
        solo_wave = []
        solo_midi = []
        for group in groups:
            # 收集每个乐器的midi和wave
            npmidis = []
            waves = []
            for inst_idx in group:
                solo_path = piece_path + str(inst_idx)
                for file_name in os.listdir(solo_path):
                    if file_name.endswith('.npy'):
                        midi = np.load(os.path.join(solo_path, file_name))
                        # (84, T)
                        npmidis.append(midi)
                    elif file_name.endswith('.wav'):
                        wave_data, _ = torchaudio.load(os.path.join(solo_path, file_name))
                        # wave_data: (1, num_samples)
                        waves.append(wave_data)
            if len(npmidis) != len(group) or len(waves) != len(group):
                raise ValueError("Mismatch in number of instruments' midis and waves.")
            # 混合
            merged_midi = reduce(midiarray_add, npmidis)    # midi长度不一定一样
            solo_midi.append(merged_midi)

            merged_wave = _merge_waves(waves)
            solo_wave.append(merged_wave)

        ensemble_midi = _stack_midis(solo_midi)  # (num_instruments, 84, T)
        ensemble_wave = _merge_waves(solo_wave) # (1, num_samples)
        # 和下面的保持一致
        ensemble_name = ensemble_name + '_' + '&'.join([str(g[0]) for g in groups])
        yield ensemble_name, ensemble_midi, ensemble_wave


def amt_urmp(model, output_folder, mix=2, normalize=True):
    output_path = os.path.join(output_folder, 'URMP_eval')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for name, midi, wave in urmp_folder_iterator(mix=mix):
        print(f"Processing {name}")
        wave.unsqueeze_(0)  # (1, num_samples)
        onset, note, emb = model(wave)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # D维归一化
        onset = onset.cpu().numpy()[0]  # [F, T]
        note = note.cpu().numpy()[0]  # [F, T]
        emb = emb.cpu().numpy()[0]  # [D, F, T]
        if normalize:
            onset = onset / np.max(onset)  # [F, T]
            note = note / np.max(note)  # [F, T]
        # padding midi in time axis
        if midi.shape[-1] < note.shape[-1]:
            pad_width = note.shape[-1] - midi.shape[-1]
            midi = np.pad(midi, ((0, 0), (0, 0), (0, pad_width)))
        elif midi.shape[-1] > note.shape[-1]:
            midi = midi[:, :, :note.shape[-1]]
        np.savez(
            os.path.join(output_path, f"{name}.npz"),
            emb=emb,
            frame=note,
            onset=onset,
            midi=midi
        )


### 预处理和保存结果，只保留emb, onset, note和midi，单独文件保存

def amt_mix(model, folders, normalize=True):
    """
    mix audios and get AMT results
    folders: ["music@1", "music@2", ...], each folder contains .npy and .wav
    """
    # get all the wav & midi
    waveforms = []
    midis = []
    for folder in folders:
        filename = os.listdir(folder)[0]
        path = os.path.join(folder, filename)[:-3]    # 去掉后缀
        _wave, _ = torchaudio.load(path+'wav') # wave: [1, T], sr = 22050
        waveforms.append(_wave)
        midi = np.load(path+"npy")
        midis.append(midi)
    # mix wav
    mix_wave = _merge_waves(waveforms)
    mix_wave.unsqueeze_(0)  # [1, 1, T]
    # model output
    onset, note, emb = model(mix_wave)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # D维归一化
    onset = onset.cpu().numpy()[0]  # [F, T]
    note = note.cpu().numpy()[0]  # [F, T]
    emb = emb.cpu().numpy()[0]  # [D, F, T]
    if normalize:
        onset = onset / np.max(onset)  # [F, T]
        note = note / np.max(note)  # [F, T]
    # padding midi in time axis
    midis = _stack_midis(midis)  # [len(folders), F, T]
    if midis.shape[-1] < note.shape[-1]:
        pad_width = note.shape[-1] - midis.shape[-1]
        midis = np.pad(midis, ((0, 0), (0, 0), (0, pad_width)))
    elif midis.shape[-1] > note.shape[-1]:
        midis = midis[:, :, :note.shape[-1]]
    return onset, note, emb, midis

def amt_mix_dataset(model, dataset_folder, mix, output_folder):
    """
    mix audios in dataset and get AMT results
    dataset_folder: "BACH10_processed", in which each ensemble contains 4 instruments
    """
    folder_name = os.path.basename(dataset_folder)
    output_folder_name = folder_name.split("_")[0] + "_eval"
    output_path = os.path.join(output_folder, output_folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    folders = os.listdir(dataset_folder)
    for folder in folders:
        # find folders that end with 0
        # e.g. "music@0"
        if not os.path.isdir(os.path.join(dataset_folder, folder)):
            continue
        if not folder.endswith("0"):
            continue
        piece_name = folder[:-2]
        # find all folders that start with the same piece name
        # e.g. "music@1", "music@2", "music@3", "music@4"
        parts = []
        for f in folders:
            path = os.path.join(dataset_folder, f)
            if not os.path.isdir(path):
                continue
            if f.startswith(piece_name) and (not f.endswith("0")):
                parts.append(path)
        # choose {mix} folders to mix
        if len(parts) < mix:
            print(f"Not enough parts for {piece_name}, skip")
            continue
        for selected_parts in combinations(parts, mix):
            processing_piece_name = piece_name + "_" + '&'.join([part[-1] for part in selected_parts])
            print(f"Processing {processing_piece_name}")
            onset, note, emb, midis = amt_mix(model, selected_parts)
            # save emb, note, midis
            np.savez(
                os.path.join(output_path, f"{processing_piece_name}.npz"),
                emb=emb,
                frame=note,
                onset=onset,
                midi=midis
            )


### 计算帧级&音符级的评价指标
from instrument_agnostic_eval_utils import frame_eval, find_best_threshold, find_best_onset_threshold, _freqmap
from itertools import permutations
from utils.postprocess import cluster_notes
from utils.midiarray import notes2numpy

def eval_sep_note(frame: np.ndarray, onset: np.ndarray, emb: np.ndarray, midi: np.ndarray, frame_thresh = 0.5, onset_thresh = 0.4, s_per_frame = 256 / 22050, freqmap = _freqmap):
    if midi.ndim != 3:
        raise ValueError("midi should be of shape (num_instruments, F, T)")
    n_clusters = midi.shape[0]
    est_notes = cluster_notes(
        frame, onset, emb, n_clusters,
        midi_offset = 0,
        frame_thresh=frame_thresh,
        onset_thresh=onset_thresh,
        min_note_len=7.5,
        infer_onsets=True,
        melodia_trick=True,
        neighbor_trick=False,
        energy_tol = 11,
    )

    midiarrays = [
        notes2numpy(
            class_n,
            note_range=(0, 83),
            max_time_steps=frame.shape[-1],
            need_onset=False,
            need_velocity=False,
        ) for class_n in est_notes
    ]
    midi_clip = np.clip(midi, 0, 1)
    # 补零使midi_clip与frame长度一致
    if midi_clip.shape[-1] < frame.shape[-1]:
        pad_width = frame.shape[-1] - midi_clip.shape[-1]
        midi_clip = np.pad(midi_clip, ((0, 0), (0, 0), (0, pad_width)))
    elif midi_clip.shape[-1] > frame.shape[-1]:
        midi_clip = midi_clip[:, :, :frame.shape[-1]]

    # 用PIT找到最优的对应顺序
    lossmin = float("inf")
    best_order = None
    for order in permutations(range(len(midiarrays))):
        permuted = [midiarrays[i] for i in order]
        loss = np.sum(np.power(midi_clip - np.stack(permuted, axis=0), 2))
        if loss < lossmin:
            lossmin = loss
            best_order = order

    if best_order is None:
        raise ValueError("No valid permutation found for evaluation.")

    order = list(best_order)

    actual_notes = array2notes(midi)
    ps = []
    rs = []
    fs = []
    avg_overlaps = []
    for i in range(n_clusters):
        est_note_events = est_notes[order[i]]
        gt_note_events = actual_notes[i]

        est_note_events.sort(key=lambda x: x[0])  # 按onset排序
        gt_note_events.sort(key=lambda x: x[0])  # 按onset排序

        # 转换为mir_eval所需格式
        est_intervals = np.array([[onset * s_per_frame, offset * s_per_frame] for onset, offset, _, _ in est_note_events])
        est_pitches = np.array([freqmap[note] for _, _, note, _ in est_note_events])
        ref_intervals = np.array([[onset * s_per_frame, offset * s_per_frame] for onset, offset, _ in gt_note_events])
        ref_pitches = np.array([freqmap[note] for _, _, note in gt_note_events])
        p, r, f, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=0.05,
            # 不关心offset
            offset_ratio=None # type: ignore
        )
        ps.append(p)
        rs.append(r)
        fs.append(f)
        avg_overlaps.append(avg_overlap_ratio)

    P = np.mean(ps)
    R = np.mean(rs)
    F = np.mean(fs)
    AvgOverlap = np.mean(avg_overlaps)
    return P, R, F, AvgOverlap


def evaluate_sep_note_dataset(npyfolder, frame_thresh = 0.5, onset_thresh = 0.4, mix = 2, log = True):
    """
    在给定阈值的时候，评估分离效果
    """
    ps = []
    rs = []
    fs = []
    overlaps = []
    for f in os.listdir(npyfolder):
        if not ifNmix(f, mix):
            continue
        npy_path = os.path.join(npyfolder, f)
        result = np.load(npy_path)
        emb = result['emb']  # (D, freqs, times)
        note = result['frame']  # (freqs, times)
        midi = result['midi']  # (mix, freqs, times)
        onset = result['onset']  # (freqs, times)
        p, r, f, overlap = eval_sep_note(
            frame=note,
            onset=onset,
            emb=emb,
            midi=midi,
            frame_thresh=frame_thresh,
            onset_thresh=onset_thresh,
        )
        ps.append(p)
        rs.append(r)
        fs.append(f)
        overlaps.append(overlap)

    P = np.mean(ps)
    R = np.mean(rs)
    F = np.mean(fs)
    OVERLAP = np.mean(overlaps)

    if log:
        print("=== Separation Note-level Evaluation ===")
        print(f"| Precision | Recall | F | Overlap |")
        print(f"| {P:.5f} | {R:.5f} | {F:.5f} | {OVERLAP:.5f} |")

    return P, R, F, OVERLAP