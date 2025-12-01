"""Dashboard UI components."""
from .header import render_header
from .power_panel import render_power_panel
from .detection_panel import render_detection_panel
from .state_machine import render_state_machine_panel
from .audio_panel import render_audio_panel
from .spectrogram_panel import render_spectrogram_panel
from .event_log import render_event_log

__all__ = [
    'render_header',
    'render_power_panel',
    'render_detection_panel',
    'render_state_machine_panel',
    'render_audio_panel',
    'render_spectrogram_panel',
    'render_event_log',
]
