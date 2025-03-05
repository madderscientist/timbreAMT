import numpy as np
from scipy.stats import truncnorm

OCTAVE_WEIGHT = [1, 3, 5, 6, 5, 3, 1]
LEN_RANGE = (3, 120)
LEN_MEAN = 24   # 在17.4ms一格时为144bpm的四分音符
SIGMA_SCALE = 2   # 方差遵循2sigma准则 保证绝大多数数据在范围内
CHORDS = [4, 5, 7, 12]  # 泛音会重叠的

class Notes:
    def __init__(self, octave_weight = OCTAVE_WEIGHT, len_range = LEN_RANGE, len_mean = LEN_MEAN, len_sigma = -1):
        self.octave_weight = list(octave_weight)
        self.note_num = len(octave_weight) * 12
        self.len_range = tuple(len_range)  # 每个音符的长度范围
        self.len_mean = len_mean    # 每个音符的平均长度
        self.len_sigma = len_sigma if len_sigma > 0 else (len_mean - len_range[0]) / SIGMA_SCALE
        self.init_len_dist()
        self.notes = []

    def init_len_dist(self):
        """
        用截断正态分布生成音符长度
        采样器存储于self.len_dist
        """
        self.len_dist = truncnorm(
            # 加上0.4因为最后会用round，不加会导致最边上两个点取值概率偏小
            (self.len_range[0] - 0.4 - self.len_mean) / self.len_sigma,
            (self.len_range[1] + 0.4 - self.len_mean) / self.len_sigma,
            loc=self.len_mean,
            scale=self.len_sigma
        )

    def set_mean_len_from_bpm(self, bpm=144, ms_per_grid=17.4):
        """
        根据bpm和每个格子的时间长度设置音符平均长度
        """
        quarter = 60000 / bpm / ms_per_grid
        self.len_mean = int(round(quarter))
        self.len_range = (int(round(quarter / 8)), int(round(quarter * 5)))
        self.len_sigma = (self.len_mean - self.len_range[0]) / SIGMA_SCALE
        self.init_len_dist()
    
    def current_bpm(self, ms_per_grid=17.4):
        """
        返回当前的bpm
        """
        return 60000 / self.len_mean / ms_per_grid

    def new(self, octave_weight = None):
        """
        添加新的音符序列
        """
        if octave_weight is None:
            octave_weight = self.octave_weight
        note_num = len(octave_weight) * 12
        for noteid in range(note_num):
            octave = noteid // 12
            weight = octave_weight[octave]
            self.notes += [noteid] * weight
        np.random.shuffle(self.notes)
    
    def nearest(self, note, future_len = 10):
        """
        把剩下future_len中最近的音符放到最后；此机制的设置目的是让音符有一定的主线
        note: 需要靠近的音符的id
        future_len: 未来多少音符内找最近的音符
        """
        if future_len <= 1:
            return
        future_len = min(future_len, len(self.notes))
        distance_min = 100
        nearest_at = -1
        for i in range(future_len):
            dis = abs(self.notes[-i] - note)
            if dis < distance_min:
                distance_min = dis
                nearest_at = i
        # 交换位置
        self.notes[-nearest_at], self.notes[-1] = self.notes[-1], self.notes[-nearest_at]
    
    def aChord(self, note):
        dis = np.random.choice(CHORDS)
        dis = dis if note < self.note_num / 2 else -dis
        return note + dis

    def fetch(self, last_note = None, near_future_len = 0):
        """
        从音符序列中取出一个音符 只考虑了上一个音符，没有考虑之前的
        last_note: 上一个音符，即：(音符id，相对上一个音符末尾的偏移，音符长度)
        near_future_len: 未来多少音符内找最近的音符、
        return: (音符id，相对上一个音符末尾的偏移，音符长度)        
        """
        if len(self.notes) == 0:
            self.new()
        if last_note is None:
            last_note = (-1, 0, 0)  # 没有长度的音符
            near_future_len = 0

        self.nearest(last_note[0], near_future_len)
        note = self.notes.pop()

        length = int(round(self.len_dist.rvs()))

        offset_dist = truncnorm(
            (-last_note[2] - 0.5) / self.len_sigma, # 保证本音符不会在上一个音符前面
            # 最大空余设置成平均音长的一半了，因为这样能更密集
            (self.len_mean / 2 + 0.5) / self.len_sigma, # 0.5因为round是四舍五入，需要补偿边沿
            loc=-self.len_range[0],
            scale=self.len_sigma
        )
        offset = int(round(offset_dist.rvs()))
        return (note, offset, length)

    def putback(self, note):
        """
        将音符放回序列
        note: 音符id
        """
        self.notes.append(note)
    
    def generate(self, frames = 660, polyphonic = 0.17, near = 0.7):
        """
        生成一个音符序列
        frames: 音符序列的长度
        polyphonic: 多音符概率，即同时演奏的音符出现的概率
        near: 靠近上一个音符的概率，即音符连贯性
        return: 一个numpy二维数组，每行代表一个音高，每列代表一个时间点（时间在第二维）
        """
        ft = np.zeros((self.note_num, frames))
        polyphonic_time = []
        current_time = 0
        last = None

        def put_note(note, offset, length):
            nonlocal current_time
            note_row = ft[note]
            current_time = max(0, current_time + offset)
            if current_time >= frames:
                return False
            # 向后找到一个空位
            while current_time < frames and note_row[current_time] != 0:
                current_time += 1
            # 放置音符 如果位置太小就不放
            if current_time + length > frames:
                length = frames - current_time
                if length < (self.len_mean / 2):
                    return False
            # 是否多音符 越长越有可能 最多加0.2
            if np.random.uniform(0, 1) < polyphonic + length / self.len_range[1] * 0.2:
                polyphonic_time.append((current_time, note))
            # 正式填充
            note_row[current_time] = 2
            note_row[current_time + 1 : min(frames, current_time + length)] = 1
            current_time += length
            return True

        def a_note(last_note, sepcifyOffset = None, sepcifyLength = None, sepcifyNote = None):
            nonlocal current_time
            now = self.fetch(
                last_note,
                near_future_len = 9 if np.random.uniform(0, 1) < near else 0
            )
            note, offset, length = now
            if sepcifyOffset != None:
                offset = sepcifyOffset
            if sepcifyLength != None:
                length = sepcifyLength
            if sepcifyNote != None:
                self.putback(note)
                note = sepcifyNote

            if put_note(note, offset, length):
                return True, now
            else:
                if sepcifyNote == None:
                    self.putback(note)
                return False, None

        while current_time < frames:
            next, last = a_note(last)
            if not next:
                break
        
        # 处理polyphonic_time
        i = 0
        while i < len(polyphonic_time):
            current_time, note = polyphonic_time[i]

            if np.random.uniform(0, 1) < polyphonic:
                polyphonic_time.append(polyphonic_time[i])

            specify_note = None
            if np.random.uniform(0, 1) < 0.85:
                # 85%的概率是和弦
                specify_note = self.aChord(note)

            a_note(None, sepcifyOffset = 0, sepcifyNote = specify_note)
            i += 1
            
        return ft