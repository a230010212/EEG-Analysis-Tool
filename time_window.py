"""Streamlit time-window control with range dragging, touch support, and handles."""
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components


_ASSETS = Path(__file__).parent / 'ui' / 'time_window'
_component = components.component(
    'eeg_time_window',
    html=(_ASSETS / 'index.html').read_text(encoding='utf-8'),
    css=(_ASSETS / 'style.css').read_text(encoding='utf-8'),
    js=(_ASSETS / 'control.js').read_text(encoding='utf-8'),
)


def time_window(total_duration, *, key):
    """Return (start, end) in seconds; only commit pointer changes on release."""
    initial = [0.0, min(30.0, total_duration)]
    selection = st.session_state.get(key, {}).get('selection', initial)
    # Keep previously saved fractional selections from leaking into the UI.
    selection = [float(int(float(value) + 0.5)) for value in selection]
    try:
        result = _component(
            key=key,
        data={'total': total_duration, 'selection': selection, 'step': 1},
            default={'selection': initial},
            on_selection_change=lambda: None,
        )
        if hasattr(result, 'selection'):
            start, end = result.selection
            return float(start), float(end)
        if isinstance(result, dict) and 'selection' in result:
            start, end = result['selection']
            return float(start), float(end)
    except Exception:
        pass
    return float(selection[0]), float(selection[1])
