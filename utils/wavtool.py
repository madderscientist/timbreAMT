import wave
import numpy as np

def cutWave(input_wavefile: str, output_wavefile: str, start_time: float, end_time: float, mono: bool = False):
    """
    裁剪wave文件
    input_wavefile: 输入的wave文件
    output_wavefile: 输出的wave文件
    start_time: 裁剪的开始时间（s）
    end_time: 裁剪的结束时间（s）
    mono: 是否转换为单声道
    """
    with wave.open(input_wavefile, 'rb') as wave_read:
        params = wave_read.getparams()
        framerate = wave_read.getframerate()
        sampwidth = wave_read.getsampwidth()
        start_frame = int(start_time * framerate)
        end_frame = int(end_time * framerate)
        wave_read.setpos(start_frame)
        frames = wave_read.readframes(end_frame - start_frame)
        with wave.open(output_wavefile, 'wb') as wave_write:
            wave_write.setparams(params)
            if mono:
                wave_write.setnchannels(1)
                dtype = np.int8 if sampwidth == 1 else np.int16 if sampwidth == 2 else np.int32
                audio_data = np.frombuffer(frames, dtype=dtype)
                # 将双声道数据转换为单声道数据
                # 是一个一维数组，两个采样点为一帧
                mono_data = ((audio_data[::2] + audio_data[1::2]) / 2).astype(dtype)
                wave_write.writeframes(mono_data.tobytes())
            else:
                wave_write.writeframes(frames)

def waveInfo(wavefile: str):
    """获取wave文件的信息"""
    with wave.open(wavefile, 'rb') as file:
        nFrames = file.getnframes()
        frameRate = file.getframerate()
        sampleWidth = file.getsampwidth()
        print("Sample rate:", frameRate)
        print("Channels:", file.getnchannels())
        print("Sample width:", sampleWidth)
        print("Number of frames:", nFrames)
        print("Duration (s):", nFrames / frameRate)
        print("data num:", int(len(file.readframes(nFrames))/sampleWidth))