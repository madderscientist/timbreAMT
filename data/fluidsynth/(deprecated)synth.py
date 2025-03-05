""" 已废弃
废弃愿意：用命令行调用太慢了，每次都要重新加载soundfont
模仿的是https://github.com/bzamecnik/midi2audio
"""
import os
import subprocess

ROOT = './fluidsynth'
DEFAULT_SOUND_FONT = os.path.join(ROOT, 'MS Basic.sf3')
DEFAULT_FLUIDSYNTH = os.path.join(ROOT, 'fluidsynth')
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_GAIN = 1.0

class FluidSynth():
    def __init__(self, sound_font=DEFAULT_SOUND_FONT, sample_rate=DEFAULT_SAMPLE_RATE, gain=DEFAULT_GAIN, fluidsynth = DEFAULT_FLUIDSYNTH):
        self.sample_rate = sample_rate
        self.sound_font = os.path.expanduser(sound_font)
        self.gain = gain
        self.fluidsynth = fluidsynth

    """
    Convert a MIDI file to an audio file.

    Args:
        midi_file: The path to the MIDI file.
        audio_file: The path to the audio file.
        verbose: If True, print the output of the command. If False, suppress the output.
    
    Returns:
        None
    """
    def midi_to_audio(self, midi_file: str, audio_file: str, verbose=True):
        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
        subprocess.call(
            [self.fluidsynth, '-ni', '-g', str(self.gain), self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)], 
            stdout=stdout, 
        )

    def play_midi(self, midi_file):
        subprocess.call(['fluidsynth', '-i', '-g', str(self.gain), self.sound_font, midi_file, '-r', str(self.sample_rate)])

# 使用举例
if __name__ == '__main__':
    S = FluidSynth()
    S.midi_to_audio('孤独な巡礼simple.mid', 'test.mp3', False)