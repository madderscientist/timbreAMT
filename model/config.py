from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CQTConfig:
    fs: int = 22050              # Sample rate of input audio
    bins_per_octave: int = 36    # Number of frequency bins per octave
    fmin: float = 32.07955284    # Minimum frequency
    octaves: int = 8             # Number of octaves
    hop: int = 256               # Hop size (frame shift) recommended to be 2^o
    filter_scale: int = 1        # Filter scale factor

    # support dict-like access
    def __getitem__(self, key):
        return getattr(self, key)

@dataclass(frozen=True)
class Config:
    MIDI_MIN: int = 24           # Minimum MIDI pitch
    MIDI_MAX: int = 107          # Maximum MIDI pitch

    CQT: CQTConfig = field(default_factory=CQTConfig)  # CQT parameter configuration
    s_per_frame: float = field(init=False)             # Seconds per frame

    def __post_init__(self):
        object.__setattr__(self, "s_per_frame", self.CQT.hop / self.CQT.fs)

    # support dict-like access
    def __getitem__(self, key):
        return getattr(self, key)

CONFIG = Config()