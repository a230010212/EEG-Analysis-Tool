export default function({parentElement, data, setStateValue}) {
  const root = parentElement.querySelector('.time-window');
  const track = root.querySelector('.track');
  const selection = root.querySelector('.selection');
  const startHandle = root.querySelector('.start');
  const endHandle = root.querySelector('.end');
  const startInput = root.querySelector('.start-input');
  const endInput = root.querySelector('.end-input');
  const previous = root.querySelector('.previous');
  const next = root.querySelector('.next');
  const total = data.total;
  const step = data.step;
  const minimum = Math.min(step, total);
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const snap = v => Math.round(v / step) * step;
  const format = v => Number(v.toFixed(3)).toString();
  let start = clamp(data.selection[0], 0, total - minimum);
  let end = clamp(data.selection[1], start + minimum, total);
  let drag = null;
  const controller = new AbortController();
  const on = (element, name, handler) => element.addEventListener(name, handler, {signal: controller.signal});

  function render() {
    selection.style.left = `${start / total * 100}%`;
    selection.style.width = `${(end - start) / total * 100}%`;
    startHandle.style.left = `${start / total * 100}%`;
    endHandle.style.left = `${end / total * 100}%`;
    root.querySelector('.duration').textContent = `已選 ${format(end - start)} 秒`;
    root.querySelector('.total').textContent = `${format(total)} 秒`;
    startInput.value = format(start);
    endInput.value = format(end);
    startInput.max = format(end - minimum);
    endInput.min = format(start + minimum);
    endInput.max = format(total);
    for (const [element, min, max, value, text] of [
      [startHandle, 0, end - minimum, start, `${format(start)} 秒`],
      [endHandle, start + minimum, total, end, `${format(end)} 秒`],
      [selection, 0, total - (end - start), start, `${format(start)} 至 ${format(end)} 秒，長度 ${format(end - start)} 秒`],
      [root.querySelector('.drag-helper'), 0, total - (end - start), start, `${format(start)} 至 ${format(end)} 秒，長度 ${format(end - start)} 秒`],
    ]) {
      element.setAttribute('aria-valuemin', min);
      element.setAttribute('aria-valuemax', max);
      element.setAttribute('aria-valuenow', value);
      element.setAttribute('aria-valuetext', text);
    }
    previous.disabled = start <= 1e-8;
    next.disabled = end >= total - 1e-8;
    selection.querySelector('.grip').style.visibility = (end - start) / total * track.clientWidth < 44 ? 'hidden' : 'visible';
  }

  function commit() {
    setStateValue('selection', [Number(start.toFixed(6)), Number(end.toFixed(6))]);
  }

  function move(delta, baseStart = start, baseEnd = end) {
    const offset = clamp(delta, -baseStart, total - baseEnd);
    start = baseStart + offset;
    end = baseEnd + offset;
  }

  on(root, 'pointerdown', event => {
    const target = event.target.closest('[data-part]');
    if (!target || event.button !== 0 || drag) return;
    event.preventDefault();
    target.focus({preventScroll: true});
    target.setPointerCapture(event.pointerId);
    drag = {part: target.dataset.part, x: event.clientX, start, end,
            width: track.getBoundingClientRect().width, id: event.pointerId, target};
    if (drag.part === 'move') selection.classList.add('dragging');
  });

  on(root, 'pointermove', event => {
    if (!drag || event.pointerId !== drag.id) return;
    const delta = snap((event.clientX - drag.x) / drag.width * total);
    if (drag.part === 'move') move(delta, drag.start, drag.end);
    if (drag.part === 'start') start = clamp(drag.start + delta, 0, end - minimum);
    if (drag.part === 'end') end = clamp(drag.end + delta, start + minimum, total);
    render();
  });

  function finish(event) {
    if (!drag || event.pointerId !== drag.id) return;
    if (event.type === 'pointercancel') {
      start = drag.start;
      end = drag.end;
    }
    drag = null;
    selection.classList.remove('dragging');
    render();
    commit();
  }
  on(root, 'pointerup', finish);
  on(root, 'pointercancel', finish);
  on(root, 'lostpointercapture', finish);

  on(root, 'keydown', event => {
    const part = event.target.dataset.part;
    if (!part || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const forward = ['ArrowRight', 'ArrowUp'].includes(event.key);
    let delta = (forward ? 1 : -1) * step * (event.shiftKey ? 10 : 1);
    if (part === 'move') {
      if (event.key === 'Home') delta = -start;
      if (event.key === 'End') delta = total - end;
      move(delta);
    } else if (part === 'start') {
      start = event.key === 'Home' ? 0 : event.key === 'End' ? end - minimum : clamp(start + delta, 0, end - minimum);
    } else {
      end = event.key === 'Home' ? start + minimum : event.key === 'End' ? total : clamp(end + delta, start + minimum, total);
    }
    render();
    commit();
  });

  on(startInput, 'change', () => {
    if (startInput.value !== '' && Number.isFinite(startInput.valueAsNumber)) start = clamp(startInput.valueAsNumber, 0, end - minimum);
    render(); commit();
  });
  on(endInput, 'change', () => {
    if (endInput.value !== '' && Number.isFinite(endInput.valueAsNumber)) end = clamp(endInput.valueAsNumber, start + minimum, total);
    render(); commit();
  });
  on(previous, 'click', () => { move(-(end - start)); render(); commit(); });
  on(next, 'click', () => { move(end - start); render(); commit(); });
  const observer = new ResizeObserver(render);
  observer.observe(track);
  render();
  return () => { controller.abort(); observer.disconnect(); };
}
