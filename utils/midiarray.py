"""
用于txt<-->MIDI<-->numpy互转，函数名大多包含"2"(to)
0表示没有音符
1表示有音符
2表示onset
忽略了响度信息
乐器事件不在头文件，而是每个有音符的音轨
"""

from typing import Any, Union, List, Tuple, Optional
import mido
import numpy as np

# 本脚本产生的所有midi的标准配置
# 不关心节奏，只关心音符
TEMPO = 500000  # 120bpm
TICKS_PER_BEAT = 480

def midi2numpy(
        midi_file: Union[str, mido.MidiFile],
        time_step: float = 0.010,
        note_range: Tuple[int, int] = (24, 107),
        time_first: bool = False,
        track_separate: bool = False
    ) -> np.ndarray:
    """
    midi_file: midi文件路径 或者 mido.MidiFile对象
    time_step: 时间步长，单位秒
    note_range: 音符范围
    time_first: 时间是否是第一个维度
    track_separate: 是否分开每个音轨
    return: numpy数组, shape=(num_tracks, num_notes, num_time_steps) if track_separate else (num_notes, num_time_steps)
    """
    mid = mido.MidiFile(midi_file) if isinstance(midi_file, str) else midi_file
    num_notes = note_range[1] - note_range[0] + 1
    n_beat_per_us_ticks = 1e-6 / time_step / mid.ticks_per_beat # 再乘一个(us/beat)就得到了(n/tick)

    # 获取tempo和总时长 之所以不用提供的length属性是因为它也是遍历的，我这里遍历还能获取tempo、提取note事件
    all_tick_num = 0
    # [开始tick, 结束tick，在此之前的时间，npt(只要乘上tick就是格序号)]
    n_per_ticks = [[0, 0, 0, mido.midifiles.midifiles.DEFAULT_TEMPO * n_beat_per_us_ticks]]
    tracks = []
    for track in mid.tracks:
        track_tick = 0
        track_tick_note = 0
        track_msg = []
        has_note = False
        for m in track:
            track_tick += m.time
            if m.type == 'set_tempo':
                # -1的项后面填充
                n_per_ticks.append([track_tick, -1, -1, m.tempo * n_beat_per_us_ticks])
            elif m.type[:4] == 'note':
                has_note = True
                track_tick_note = track_tick
                track_msg.append(m.copy(time=track_tick))
        if track_tick_note > all_tick_num:
            all_tick_num = track_tick
        if has_note:
            tracks.append(track_msg)

    num_time_steps = 0

    n_per_ticks.sort()
    for i in range(1, len(n_per_ticks)):
        # 假设最后一个tempo_change之后还有note
        num_time_steps += (n_per_ticks[i][0] - n_per_ticks[i-1][0]) * n_per_ticks[i-1][3]
        n_per_ticks[i-1][1] = n_per_ticks[i][0] # 结束tick
        n_per_ticks[i][2] = num_time_steps      # 在此之前的时间
    num_time_steps += (all_tick_num - n_per_ticks[-1][0]) * n_per_ticks[-1][3]
    num_time_steps = int(np.ceil(num_time_steps))   # 格子数
    n_per_ticks[-1][1] = float('inf')

    def track2numpy(track):
        tempo_idx = 0
        begin_time, valid_time, time_bef, npt = n_per_ticks[0]
        piano_roll = np.zeros((num_notes, num_time_steps), dtype=np.int8)
        active_notes = np.zeros(num_notes, dtype=int)
        
        for m in track:
            # 更新tempo
            while m.time >= valid_time:
                tempo_idx += 1
                begin_time, valid_time, time_bef, npt = n_per_ticks[tempo_idx]

            note = m.note - note_range[0]
            if note < 0 or note >= num_notes:
                continue

            time = int(round(time_bef + (m.time - begin_time) * npt))
            # if time >= num_time_steps:
            #     break 不会发生

            if active_notes[note] > 0:      # 有音符在响 赋值为1表示中间过程
                piano_roll[note, active_notes[note]:time] = 1

            if m.type == 'note_off' or m.velocity == 0:
                active_notes[note] = 0
            else:
                piano_roll[note, time] = 2
                active_notes[note] = time + 1   # 从1开始哦
       
        if time_first:
            return piano_roll.T
        return piano_roll

    if track_separate:
        return np.stack([track2numpy(track) for track in tracks], axis=0)
    else:
        msgs = []
        for t in tracks:
            msgs.extend(t)
        msgs.sort(key=lambda msg: msg.time)
        return track2numpy(msgs)


def numpy2midi(
        arr: np.ndarray,
        time_step: float = 0.010,
        time_first: bool = False,
        min_note: int = 24,
        random: bool = False,
        instrument: Union[int, List[int]] = 0
    ) -> mido.MidiFile:
    """
    arr: numpy数组, 第一维tracks可省略, shape=(tracks, num_time_steps, num_notes) if time_first else (tracks, num_notes, num_time_steps)
    time_step: 时间步长，单位秒
    time_first: 时间是否是第一个维度
    min_note: 最小音符的midi值 默认24为C1
    random: 是否为midi事件添加随机性
    instrument: 乐器的midi编号 可以是一个数组表示每一维度的乐器 缺少的会用最后一个作为后面音轨的乐器，多则忽略
    """
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if time_first:  # 时间是最后一个维度
        arr = np.transpose(arr, (0, 2, 1))

    if isinstance(instrument, int):
        instrument = [instrument]

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    scale = mid.ticks_per_beat / (TEMPO * 1e-6) * time_step
    num_notes = arr.shape[2]

    headTrack = mido.MidiTrack()
    mid.tracks.append(headTrack)
    headTrack.append(mido.MetaMessage('track_name', name='head', time=0))
    headTrack.append(mido.MetaMessage('set_tempo', tempo=TEMPO, time=0))
    headTrack.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    headTrack.append(mido.MetaMessage('end_of_track', time=0))

    def numpy2track(arr, track = None):
        if track is None:
            track = mido.MidiTrack()
        events = []
        activates = np.zeros(num_notes, dtype=int) # 理论上可以一直noteon不noteoff，但是有的软件不会把这个音断开
        for (note, t), state in np.ndenumerate(arr):
            if random:  # 随机化 可以左右偏一些 因为量化到格子上也是round得到的，损失了一些信息，这里补回来，使数据更真实
                t = max(0, t + np.random.uniform(-0.49, 0.49))
            if state < 0.66:    # 0
                if activates[note] > 0:
                    events.append((t, 0, note))
                    activates[note] = 0
            elif state > 1.34:  # 2
                if activates[note] == 0:
                    events.append((t, 1, note))
                elif activates[note] == 1:   # 之前已经激活，先关了再开
                    events.append((t, 0, note))
                    events.append((t, 1, note))
                activates[note] = 2
            else:               # 1
                if activates[note] == 0:    # 理论上不该出现，实际上开头没必要总是2
                    events.append((t, 1, note))
                    activates[note] = 2
                else:
                    activates[note] = 1

        for note, state in enumerate(activates):
            if state != 0:
                events.append((arr.shape[1], 0, note))

        events.sort()

        last_t = 0
        for t, state, note in events:
            time = int(round((t - last_t) * scale))
            midi_note = note + min_note
            if state == 0:
                track.append(mido.Message('note_off', note=midi_note, velocity=0, time=time))
            else:
                track.append(mido.Message('note_on', note=midi_note, velocity=100, time=time))
            last_t = t
        return track

    instr_id = 0
    for piano_roll in arr:
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=instrument[instr_id], channel=len(mid.tracks)-1, time=0))
        mid.tracks.append(numpy2track(piano_roll, track))
        instr_id = instr_id + 1
        if instr_id >= len(instrument):
            instr_id = len(instrument) - 1

    return mid


def midiInstruments(
        midi_file: Union[str, mido.MidiFile]
    ) -> List[int]:
    """
    获取midi文件中使用的乐器编号
    midi_file: midi文件路径 或者 mido.MidiFile对象
    return: 乐器编号列表 如果某个音轨没有乐器事件则设置为0
    """
    mid = mido.MidiFile(midi_file) if isinstance(midi_file, str) else midi_file
    instruments = []
    for track in mid.tracks:
        has_instrument = False
        for m in track:
            if m.type == 'program_change':
                instruments.append(m.program)
                has_instrument = True
                break
        if not has_instrument:
            instruments.append(0)  # 如果没有找到乐器，默认添加0
    return instruments


def annotation2midi(
        path: str,
        cols: list = ["note", "freq", "name", "onset", "dur", "offset"],
        row_offset: int = 0,
        sep: Optional[str] = None,
        time_unit: float = 1.,
        instrument: int = 0
    ) -> mido.MidiFile:
    """
    将文本标注转换为midi
    path: str 文件路径
    cols: list 每个元素是一个列名，对应的列名在text中的位置，note和freq必须有一个（优先note）、onset必须有、dur和offset必须有一个（优先offset）
    row_offset: int 行的偏移量
    sep: str 分隔符
    time_unit: float 相对于秒的时间单位
    instrument: int 乐器的midi编号
    return: mido.MidiFile
    """
    # 检查参数
    if "onset" not in cols:
        raise ValueError("onset column is required")

    if "note" in cols:
        note_mod = 0
    elif "freq" in cols:
        note_mod = 1
    elif "name" in cols:
        note_mod = 2
    else:
        raise ValueError("note or freq or name column is required")

    if "offset" in cols:
        use_dur = False
    elif "dur" in cols:
        use_dur = True
    else:
        raise ValueError("offset or dur column is required")

    with open(path, 'r') as f:
        lines = f.readlines()

    events = []
    for line in lines[row_offset:]:
        items = line.split(sep)
        freq = -1
        note = -999
        name = ''
        onset = -1
        offset = -1
        dur = -1
        for i, item in enumerate(items):
            if item == '':
                continue
            if i >= len(cols):
                break
            col_name = cols[i]
            item = item.strip()

            if col_name == "note":
                note = int(item)
            elif col_name == "freq":
                freq = float(item)
            elif col_name == "name":
                name = item
            elif col_name == "onset":
                onset = float(item)
            elif col_name == "offset":
                offset = float(item)
            elif col_name == "dur":
                dur = float(item)

        if note_mod == 1:
            note = hz2note(freq)
        elif note_mod == 2:
            note = name2note(name)
        if use_dur:
            offset = onset + dur

        if offset == -1 or note == -999 or onset == -1:
            print(f"Warning: Invalid note or onset or offset in line: {line.strip()}")
            continue
        events.append((onset, 1, note))
        events.append((offset, 0, note))

    events.sort()

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)   # 默认一个四分音符480ticks
    scale = mid.ticks_per_beat / (TEMPO * 1e-6) * time_unit

    headTrack = mido.MidiTrack()
    mid.tracks.append(headTrack)
    headTrack.append(mido.MetaMessage('track_name', name='head', time=0))
    headTrack.append(mido.MetaMessage('set_tempo', tempo=TEMPO, time=0))
    headTrack.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    headTrack.append(mido.MetaMessage('end_of_track', time=0))

    track = mido.MidiTrack()
    track.append(mido.Message('program_change', program=instrument, channel=0, time=0))
    last_t = 0
    for t, state, note in events:
        time = int(round((t - last_t) * scale))
        midi_note = note
        if state == 0:
            track.append(mido.Message('note_off', note=midi_note, velocity=0, time=time))
        else:
            track.append(mido.Message('note_on', note=midi_note, velocity=100, time=time))
        last_t = t
    
    mid.tracks.append(track)
    return mid


def midiarray_add(midiarry1: np.ndarray, midiarry2: np.ndarray, time_first: bool = False) -> np.ndarray:
    """
    两个有midi含义的np.ndarray相加，含义是音轨合并
    midiarray1: np.ndarray 要求只能是2维
    midiarray2: np.ndarray 要求只能是2维
    time_first: 时间是否是第一个维度
    return: MidiArray
    """
    if midiarry1.ndim != 2 or midiarry2.ndim != 2:
        raise ValueError("Both input arrays must be 2-dimensional")

    if time_first:
        midiarry1 = midiarry1.T
        midiarry2 = midiarry2.T

    max_shape = (
        max(midiarry1.shape[0], midiarry2.shape[0]),
        max(midiarry1.shape[1], midiarry2.shape[1])
    )

    result = np.zeros(max_shape, dtype=np.int8)

    result[:midiarry1.shape[0], :midiarry1.shape[1]] = midiarry1
    result[:midiarry2.shape[0], :midiarry2.shape[1]] = np.maximum(
        result[:midiarry2.shape[0], :midiarry2.shape[1]], midiarry2
    )

    if time_first:
        result = result.T

    return result


def midi_merge(midis: List[mido.MidiFile]) -> mido.MidiFile:
    """
    合并多个midi文件 要求都是本脚本生成的midi文件，这样可以保证头文件一样
    关注音符和与乐器
    midis: list of mido.MidiFile
    return: mido.MidiFile
    """
    import copy

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    headTrack = mido.MidiTrack()
    mid.tracks.append(headTrack)
    headTrack.append(mido.MetaMessage('track_name', name='head', time=0))
    headTrack.append(mido.MetaMessage('set_tempo', tempo=TEMPO, time=0))
    headTrack.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    headTrack.append(mido.MetaMessage('end_of_track', time=0))

    # 要修正事件的channel
    for midi in midis:
        tracks = midi.tracks[1:]
        for t in tracks:
            cloned = copy.deepcopy(t)
            channel = len(mid.tracks) - 1
            for e in cloned:
                if hasattr(e, 'channel'):
                    e.channel = channel
            mid.tracks.append(cloned)
    
    return mid


def hz2note(hz: float, A4=440) -> int:
    """
    将频率转换为midi编码的音符
    """
    return np.round((12 * np.log2(hz / A4)) + 69).astype(int)


def name2note(name: str) -> int:
    """
    将音符名转换为midi编码的音符
    """
    name = name.upper()
    n = name[:-1]
    octave = int(name[-1])
    
    notedist = {
        'C': 0, 'C#': 1,
        'D': 2, 'D#': 3,
        'E': 4,
        'F': 5, 'F#': 6,
        'G': 7, 'G#': 8,
        'A': 9, 'A#': 10,
        'B': 11
    }
    if n not in notedist:
        raise ValueError(f"Invalid note name: {name}")
    return 12 * octave + notedist[n] + 12


def freq_map(note_range: Tuple[int, int] = (24, 107), A4: float = 440):
    return 2 ** ((np.arange(note_range[0], note_range[1]+1) - 69) / 12) * A4


def roll2evalarray(roll: np.ndarray, freq_map: np.ndarray) -> List[np.ndarray]:
    """
    Convert a piano roll (0 or 1(greater than 0)) to an array of note values which mir_eval reqires.
    :param roll: The piano roll to convert, which has been binarized. [F, T]
    :param freq_map: The frequency map to convert the piano roll to note values. [F]
    :return: The array of note values. [np.array]*T
    """
    freq_num, time_num = roll.shape
    activate = roll.astype(bool).T # (time_num, freq_num)
    eval_array = [None] * time_num
    for i in range(time_num):
        eval_array[i] = freq_map[activate[i]]
    return eval_array


def midi_randomize(
        midifile: mido.MidiFile,
        *,
        velocity_range: Tuple[int, int] = (40, 127),
        pitch_range: Tuple[int, int] = (-1365, 1365)
    ) -> mido.MidiFile:
    """
    对midi音符的力度、音高进行随机化，产生新的midi
    修改每一个音符的响度和音高，同时施加揉音
    揉音(-8192, 8191), 偏移两个半音，和音高是线性关系（和频率不是线性）。为了让音尽量准确，限制偏移范围最大1365
    """
    import copy

    min_velocity = velocity_range[0]
    max_velocity = velocity_range[1]
    avg_velocity = (min_velocity + max_velocity) / 2
    velocity_sigma = (max_velocity - avg_velocity) / 8 # 限制游走范围为一半范围，再用2sigma准则，所以是2*2*2=8sigma

    mid = mido.MidiFile(ticks_per_beat=midifile.ticks_per_beat)

    for track in midifile.tracks:
        t = mido.MidiTrack()
        mid.tracks.append(t)

        last_velocity = avg_velocity    # 模拟随机游走
        last_vibrato_time = midifile.ticks_per_beat*10

        for msg in track:
            m = copy.deepcopy(msg)
            t.append(m)
            last_vibrato_time += m.time
            if m.type == 'note_on':
                # 强度随机化，马尔可夫关系：在上一次的基础上随机游走
                v = np.random.normal(loc=last_velocity, scale=velocity_sigma)
                # 碰壁就反弹
                if v > max_velocity:
                    v = max_velocity - (v - max_velocity)
                elif v < min_velocity:
                    v = min_velocity + (min_velocity - v)
                m.velocity = int(round(v))
                last_velocity = v
                # 音高随机化偏移 三角分布
                pitch_offset = np.random.triangular(pitch_range[0], 0, pitch_range[1])
                t.append(mido.Message('pitchwheel', pitch=int(round(pitch_offset)), channel=m.channel, time=0))
                # 震音 间隔一段时间修改一次
                if last_vibrato_time > midifile.ticks_per_beat * 2:
                    shake = np.random.triangular(-30, 0, 96)
                    shake = 0 if shake <= 0 else int(round(shake))
                    t.append(mido.Message('control_change', control=1, value=shake, channel=m.channel, time=0))
                    last_vibrato_time = 0

    return mid


def array2notes(
        arr: np.ndarray,
    ) -> List[List[Tuple[int, int, int]]]:
    """将numpy数组转换为音符列表

    Args:
        arr: numpy数组, shape=(num_tracks, num_notes, num_time_steps) or (num_notes, num_time_steps)
        time_step: float 时间步长，单位秒
    
    Returns:
        音符列表的列表, 每一项代表一个音轨, 每个音轨由多个音符组成, 每个音符由(onset, offset, note)组成，时间单位为frame index
    """
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    notes_list = []
    for piano_roll in arr:
        # piano_roll: shape=(num_notes, num_time_steps)
        notes = []
        for (note, line) in enumerate(piano_roll):
            # note: shape=(num_time_steps,)
            onset = -1
            for t, val in enumerate(line):
                if val == 2:    # onset
                    if onset >= 0:   # 上一个note没有off就先off掉
                        notes.append((onset, t, note))
                    onset = t
                elif val > 0 and onset < 0: # note on
                    onset = t
                elif val == 0 and onset >= 0:
                    notes.append((onset, t, note))
                    onset = -1
                # val == 1 and onset >= 0: 继续响着
                # val == 0 and onset < 0: 继续静音
        notes_list.append(notes)
    return notes_list


def midi2notes(
        midi_file: Union[str, mido.MidiFile],
        note_range: Tuple[int, int] = (0, 127)
    ) -> List[List[Tuple[float, float, int, float]]]:
    """将midi文件转换为音符列表

    Args:
        midi_file: midi文件路径 或者 mido.MidiFile对象
        note_range: 音符范围, midi编码
    Returns:
        音符列表的列表, 每一项代表一个音轨, 每个音轨由多个音符组成, 每个音符由(onset, offset, note, velocity)组成，时间单位为秒，velocity为[0, 1]之间的浮点数
    """
    mid = mido.MidiFile(midi_file) if isinstance(midi_file, str) else midi_file
    num_notes = note_range[1] - note_range[0] + 1
    n_beat_per_us_ticks = 1e-6 / mid.ticks_per_beat # 再乘一个(us/beat)就得到了(s/tick)

    # 获取tempo和总时长 之所以不用提供的length属性是因为它也是遍历的，我这里遍历还能获取tempo、提取note事件
    all_tick_num = 0
    # [开始tick, 结束tick，在此之前的时间，spt(只要乘上tick就是秒数)]
    n_per_ticks = [[0, 0, 0, mido.midifiles.midifiles.DEFAULT_TEMPO * n_beat_per_us_ticks]]
    tracks = []
    for track in mid.tracks:
        track_tick = 0
        track_tick_note = 0
        track_msg = []
        has_note = False
        for m in track:
            track_tick += m.time
            if m.type == 'set_tempo':
                # -1的项后面填充
                n_per_ticks.append([track_tick, -1, -1, m.tempo * n_beat_per_us_ticks])
            elif m.type[:4] == 'note':
                has_note = True
                track_tick_note = track_tick
                track_msg.append(m.copy(time=track_tick))
        if track_tick_note > all_tick_num:
            all_tick_num = track_tick
        if has_note:
            tracks.append(track_msg)

    sec = 0

    n_per_ticks.sort()
    for i in range(1, len(n_per_ticks)):
        # 假设最后一个tempo_change之后还有note
        sec += (n_per_ticks[i][0] - n_per_ticks[i-1][0]) * n_per_ticks[i-1][3]
        n_per_ticks[i-1][1] = n_per_ticks[i][0] # 结束tick
        n_per_ticks[i][2] = sec      # 在此之前的时间
    sec += (all_tick_num - n_per_ticks[-1][0]) * n_per_ticks[-1][3]
    n_per_ticks[-1][1] = float('inf')

    def track2notes(track) -> List[Tuple[float, float, int, float]]:
        tempo_idx = 0
        begin_time, valid_time, time_bef, spt = n_per_ticks[0]
        active_notes = np.zeros(num_notes, dtype=int)
        notes = []
        
        for m in track:
            # 更新tempo
            while m.time >= valid_time:
                tempo_idx += 1
                begin_time, valid_time, time_bef, spt = n_per_ticks[tempo_idx]

            note = m.note - note_range[0]
            if note < 0 or note >= num_notes:
                continue

            time = time_bef + (m.time - begin_time) * spt

            if active_notes[note] > 0:      # 有音符在响 赋值为1表示中间过程
                notes.append((active_notes[note], time, note, m.velocity / 127.0))

            if m.type == 'note_off' or m.velocity == 0:
                active_notes[note] = 0
            else:
                active_notes[note] = time

        return notes

    return [track2notes(track) for track in tracks]


def notes2midi(
        notes: List[Tuple[float, float, int, float]],
        time_step: float = 256/22050,
        instrument: int = 4
    ) -> mido.MidiFile:
    """
    将音符列表转换为midi文件

    Args:
        notes: list of dict 事件列表，每一项由以下组成元素：
            onset: float 开始时间（用[0]索引）单位: 帧
            offset: float 结束时间
            note: int 音符
            velocity: float 音符力度[0, 1]
        time_step: float 时间步长，单位秒
    
    Returns:
        mido.MidiFile: 生成的midi文件
    """
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)   # 默认一个四分音符480ticks
    scale = mid.ticks_per_beat / (TEMPO * 1e-6) * time_step
    
    headTrack = mido.MidiTrack()
    mid.tracks.append(headTrack)
    headTrack.append(mido.MetaMessage('track_name', name='head', time=0))
    headTrack.append(mido.MetaMessage('set_tempo', tempo=TEMPO, time=0))
    headTrack.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    headTrack.append(mido.MetaMessage('end_of_track', time=0))

    track = mido.MidiTrack()
    track.append(mido.Message('program_change', program=instrument, channel=0, time=0))
    mid.tracks.append(track)

    # 把音符拆分为onset和offset事件
    events = []
    for onset, offset, note, velocity in notes:
        # [frame, velocity, note]
        events.append((onset, velocity, note))
        events.append((offset, 0, note))
    events.sort()

    last_frame = 0
    for frame, velocity, note in events:
        time = int(round((frame - last_frame) * scale))
        last_frame = frame
        if velocity == 0:
            track.append(mido.Message('note_off', note=note, velocity=0, time=time))
        else:
            track.append(mido.Message('note_on', note=note, velocity=int(round(velocity*127)), time=time))
    
    return mid


def notes2numpy(
        notes: List[Tuple[int, int, int, *tuple[Any, ...]]],
        note_range: Tuple[int, int] = (24, 107),
        max_time_steps: Optional[int] = None,
        need_onset: bool = True,
        need_velocity: bool = False
    ) -> np.ndarray:
    """
    将音符列表转换为numpy数组

    Args:
        notes: list of dict 事件列表，每一项由以下组成元素：
            onset: float 开始时间（用[0]索引）单位: 帧
            offset: float 结束时间
            note: int 音符
        note_range: Tuple[int, int] 音符范围，包含首尾
        max_time_steps: Optional[int] 最大时间步长，默认为None表示自动计算
        need_onset: bool 是否需要onset信息
        need_velocity: bool 是否需要velocity信息，若需要则notes中每个音符的第四个元素为velocity值，范围[0, 1]
    ) -> np.ndarray:
    """
    if max_time_steps is None:
        max_time_steps = 0
        for onset, offset, note, *rest in notes:
            if offset > max_time_steps:
                max_time_steps = offset
    num_notes = note_range[1] - note_range[0] + 1
    piano_roll = np.zeros((num_notes, max_time_steps), dtype=np.float32)
    for onset, offset, note, *rest in notes:
        if onset >= max_time_steps:
            continue
        note_idx = note - note_range[0]
        if note_idx < 0 or note_idx >= num_notes:
            continue
        if need_velocity and len(rest) > 0:
            velocity = rest[0]
        else:
            velocity = 1.0
        offset = min(offset, max_time_steps)
        if need_onset:
            piano_roll[note_idx, onset] = velocity * 2  # onset
            piano_roll[note_idx, onset+1:offset] = velocity
        else:
            piano_roll[note_idx, onset:offset] = velocity
    return piano_roll


def output2midi(
        onset: np.ndarray,
        frame: np.ndarray,
        time_step: float = 256/22050,
        frame_threshold: float = 0.14,
        onset_threshold: float = 0.4,
        min_note_sec: float = 0.09
    ) -> mido.MidiFile:
    """
    将输出转换为midi文件
    onset: np.array onset预测结果 [F, T]
    frame: np.array frame预测结果 [F, T]
    time_step: float 时间步长，单位秒
    frame_threshold: float frame阈值
    onset_threshold: float onset阈值
    min_note_sec: float 最小音符长度，单位秒
    """
    from .postprocess import output_to_notes_polyphonic
    note_events = output_to_notes_polyphonic(
        frame, onset,
        frame_thresh = frame_threshold,
        onset_thresh = onset_threshold,
        min_note_len = min_note_sec / time_step,
        infer_onsets = True,
        melodia_trick = True,
        energy_tol = 11,
        midi_offset = 24
    )
    return notes2midi(note_events, time_step) # type: ignore