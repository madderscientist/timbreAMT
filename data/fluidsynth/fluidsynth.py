# type: ignore
"""
使用fluidsynth库合成音频
"""
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_short,
    c_uint,
    c_void_p,
    create_string_buffer,
)
# 路径管理
import os

# lib_path = os.path.join(os.path.dirname(__file__), 'fluidsynth_win', 'libfluidsynth-3.dll')
# how to build fluidsynth on linux: see README.md
lib_path = os.path.join(os.path.dirname(__file__), 'fluidsynth_linux', 'libfluidsynth.so.3.5.0')

default_sound_font = os.path.join(os.path.dirname(__file__), 'MS Basic.sf3')
_fl = CDLL(lib_path)
print("Fluidsynth library loaded.")
def cfunc(name, result, *args):
    """Build and apply a ctypes prototype complete with parameter flags"""
    if hasattr(_fl, name):
        atypes = []
        aflags = []
        for arg in args:
            atypes.append(arg[1])
            aflags.append((arg[2], arg[0]) + arg[3:])
        return CFUNCTYPE(result, *atypes)((name, _fl), tuple(aflags))
    else: # Handle Fluidsynth 1.x, 2.x, etc. API differences
        return None
    
FLUID_OK = 0
FLUID_FAILED = -1

FLUID_PLAYER_READY = 0
FLUID_PLAYER_PLAYING = 1
FLUID_PLAYER_STOPPING = 2
FLUID_PLAYER_DONE = 3

## settings
new_fluid_settings = cfunc('new_fluid_settings', c_void_p)

delete_fluid_settings = cfunc('delete_fluid_settings', None,
                              ('settings', c_void_p, 1))

fluid_settings_setstr = cfunc('fluid_settings_setstr', c_int,
                              ('settings', c_void_p, 1),
                              ('name', c_char_p, 1),
                              ('str', c_char_p, 1))

fluid_settings_setnum = cfunc('fluid_settings_setnum', c_int,
                              ('settings', c_void_p, 1),
                              ('name', c_char_p, 1),
                              ('val', c_double, 1))

fluid_settings_setint = cfunc('fluid_settings_setint', c_int,
                              ('settings', c_void_p, 1),
                              ('name', c_char_p, 1),
                              ('val', c_int, 1))

fluid_synth_sfload = cfunc('fluid_synth_sfload', c_int,
                           ('synth', c_void_p, 1),
                           ('filename', c_char_p, 1),
                           ('update_midi_presets', c_int, 1))

# 关闭所有日志级别输出
fluid_set_log_function = cfunc('fluid_set_log_function', None,
                               ('level', c_int, 1),
                               ('fun', c_void_p, 1),
                               ('data', c_void_p, 1))

## player
new_fluid_player = cfunc('new_fluid_player', c_void_p,
                         ('synth', c_void_p, 1))

delete_fluid_player = cfunc('delete_fluid_player', None,
                             ('player', c_void_p, 1))

fluid_player_add = cfunc('fluid_player_add', c_int,
                         ('player', c_void_p, 1),
                         ('filename', c_char_p, 1))

fluid_player_play = cfunc('fluid_player_play', c_int,
                          ('player', c_void_p, 1))

fluid_player_stop = cfunc('fluid_player_stop', c_int,
                          ('player', c_void_p, 1))

## fluid synth
new_fluid_synth = cfunc('new_fluid_synth', c_void_p,
                        ('settings', c_void_p, 1))

delete_fluid_synth = cfunc('delete_fluid_synth', None,
                           ('synth', c_void_p, 1))

# render
new_fluid_file_renderer = cfunc('new_fluid_file_renderer', c_void_p,
                                ('synth', c_void_p, 1))

delete_fluid_file_renderer = cfunc('delete_fluid_file_renderer', None,
                                      ('renderer', c_void_p, 1))

fluid_player_get_status = cfunc('fluid_player_get_status', c_int,
                                ('player', c_void_p, 1))

fluid_file_renderer_process_block = cfunc('fluid_file_renderer_process_block', c_int,
                                          ('render', c_void_p, 1))


class Synth:
    def __init__(self, sample_rate: int, sound_font: str = default_sound_font, gain: float = 1.0, verbose: bool = False):
        self.settings = new_fluid_settings()
        if(fluid_settings_setnum(self.settings, "synth.sample-rate".encode(), c_double(sample_rate)) != FLUID_OK):
            print("Failed to set synth.sample-rate")
            exit(1)
        if(fluid_settings_setnum(self.settings, "synth.gain".encode(), c_double(gain)) != FLUID_OK):
            print("Failed to set synth.gain")
            exit(1)        
        fluid_settings_setstr(self.settings, "player.timing-source".encode(), "sample".encode())
        fluid_settings_setint(self.settings, "synth.lock-memory".encode(), 0)
        # 关闭日志输出
        if not verbose:
            fluid_settings_setint(self.settings, "synth.verbose".encode(), 0)
            for level in range(5):
                fluid_set_log_function(level, None, None)

        self.synth = new_fluid_synth(self.settings)
        fluid_synth_sfload(self.synth, sound_font.encode(), 0)

    def __del__(self):
        delete_fluid_synth(self.synth)
        delete_fluid_settings(self.settings)
        print("Synth deleted")

    def midi2audio(self, midifile = "孤独な巡礼simple.mid", audiofile = "filename.wav"):
        # 可以临时改文件名
        if(fluid_settings_setstr(self.settings, "audio.file.name".encode(), audiofile.encode()) != FLUID_OK):
            print("Failed to set audio.file.name")
            exit(1)

        player = new_fluid_player(self.synth)
        fluid_player_add(player, midifile.encode())
        fluid_player_play(player)   # 必须start才能合成

        renderer = new_fluid_file_renderer(self.synth)   # 立即创建音频文件
        while(fluid_player_get_status(player) == FLUID_PLAYER_PLAYING):
            if(fluid_file_renderer_process_block(renderer) != FLUID_OK):
                break
        delete_fluid_file_renderer(renderer)    # 删了后才释放资源

        delete_fluid_player(player) # render后的player不能用了，因为播放完了 delete源码中包含了stop

# 使用举例
if __name__ == "__main__":
    s = Synth(22050)
    midi_dir = os.path.join(os.path.dirname(__file__), '..', 'inferMusic')
    midi_file = os.path.join(midi_dir, 'short mix.mid')
    audio_file = os.path.join(midi_dir, 'short mix.wav')
    s.midi2audio(midi_file, audio_file)