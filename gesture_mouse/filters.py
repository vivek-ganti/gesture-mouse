"""One Euro filtering + cursor mapping — pure math, no OS/tracker imports.

``OneEuro`` is the standard filter from Casiez, Roussel & Vogel, "1€ Filter:
A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
(CHI 2012). Two properties are load-bearing here:

- The smoothing factor alpha is derived from the *measured* inter-sample dt,
  never a fixed frame period (camera fps halves in dim light).
- The cutoff adapts with the filtered derivative:
  ``fc = mincutoff + beta * |dx_filtered|`` — low mincutoff kills jitter at
  rest, beta restores responsiveness at speed.

``CursorPipeline`` turns tracked frames into screen-space cursor samples:
INDEX_MCP anchor -> One Euro per axis (camera px) -> control-box
sub-rectangle mapped to the full screen -> rebase offset -> screen clamp ->
filtered speed estimate (px/s) -> pixel quantization when slow. While frozen
the pipeline keeps filtering internally and only holds the *output*; the
engine rebases on unfreeze so the accumulated divergence never becomes a
cursor jump.
"""
from __future__ import annotations

import math

from .config import Config
from .types import INDEX_MCP, CursorSample, LandmarkFrame

# Below this output speed the emitted position is rounded to whole pixels to
# kill sub-pixel shimmer at rest (~2 px/frame at 30 fps).
QUANTIZE_MAX_SPEED_PX_S: float = 60.0

# Plain low-pass cutoff for the speed estimate (Hz). Fast enough that the
# engine's <50-60 px/s gates (click latch, scroll entry) see fresh values.
_SPEED_CUTOFF_HZ: float = 4.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class OneEuro:
    """Scalar One Euro filter (Casiez CHI 2012), timestamps in milliseconds.

    The first sample passes through unchanged (primes the filter). Samples
    with non-increasing timestamps return the previous output without
    mutating state.
    """

    def __init__(self, mincutoff: float, beta: float, dcutoff: float = 1.0) -> None:
        if mincutoff <= 0.0 or dcutoff <= 0.0:
            raise ValueError("cutoff frequencies must be positive")
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: float | None = None   # last filtered value
        self._dx_prev: float = 0.0          # last filtered derivative (units/s)
        self._ts_prev: float | None = None  # ms

    @staticmethod
    def _alpha(cutoff: float, dt_s: float) -> float:
        # First-order low-pass smoothing factor for cutoff (Hz) at dt (s).
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt_s)

    def filter(self, value: float, ts_ms: float) -> float:
        if self._x_prev is None or self._ts_prev is None:
            self._x_prev = value
            self._dx_prev = 0.0
            self._ts_prev = ts_ms
            return value

        dt_s = (ts_ms - self._ts_prev) / 1000.0
        if dt_s <= 0.0:
            return self._x_prev
        self._ts_prev = ts_ms

        # Derivative of the signal, itself low-passed at dcutoff.
        dx = (value - self._x_prev) / dt_s
        a_d = self._alpha(self.dcutoff, dt_s)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        # Speed-adaptive cutoff, then low-pass the value itself.
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt_s)
        x_hat = a * value + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._ts_prev = None

    def set_mincutoff(self, mincutoff: float) -> None:
        if mincutoff <= 0.0:
            raise ValueError("mincutoff must be positive")
        self.mincutoff = mincutoff


class CursorPipeline:
    """Anchor -> filtered, mapped, rebased, quantized screen cursor."""

    def __init__(self, cfg: Config, screen_w: float, screen_h: float) -> None:
        self._cfg = cfg
        self._screen_w = float(screen_w)
        self._screen_h = float(screen_h)
        oe = cfg.one_euro
        self._fx = OneEuro(oe.mincutoff, oe.beta, oe.dcutoff)
        self._fy = OneEuro(oe.mincutoff, oe.beta, oe.dcutoff)
        # beta=0 turns One Euro into a plain measured-dt low-pass.
        self._speed_filter = OneEuro(_SPEED_CUTOFF_HZ, 0.0, oe.dcutoff)
        self._pinch_filters: dict[str, OneEuro] = {}
        self._frozen = False
        self._drag = False
        self._offset: tuple[float, float] = (0.0, 0.0)
        self._pending_rebase: tuple[float, float] | None = None
        self._last_mapped: tuple[float, float] | None = None   # pre-offset
        self._prev_internal: tuple[float, float, float] | None = None  # ts,x,y
        self._speed = 0.0
        self._last_output: tuple[float, float] = (
            self._screen_w / 2.0,
            self._screen_h / 2.0,
        )

    # -- geometry -----------------------------------------------------------

    def _clamp_x(self, v: float) -> float:
        return _clamp(v, 0.0, self._screen_w - 1.0)

    def _clamp_y(self, v: float) -> float:
        return _clamp(v, 0.0, self._screen_h - 1.0)

    # -- public API ---------------------------------------------------------

    def update(self, frame: LandmarkFrame) -> CursorSample:
        ts = frame.ts_ms

        if not frame.hand_present:
            # Hold last emitted position; a fresh speed history on reacquire
            # avoids a stale-derivative spike after long absences.
            self._speed = 0.0
            self._speed_filter.reset()
            self._prev_internal = None
            hx, hy = self._last_output
            return CursorSample(hx, hy, 0.0, self._frozen, ts)

        assert frame.landmarks is not None
        anchor = frame.landmarks[INDEX_MCP]
        fx = self._fx.filter(anchor.x, ts)
        fy = self._fy.filter(anchor.y, ts)

        # Control box in camera px, read live so hot-tuning applies mid-run.
        box = self._cfg.control_box
        bx = box.x * frame.img_w
        by = box.y * frame.img_h
        bw = max(box.w * frame.img_w, 1e-6)
        bh = max(box.h * frame.img_h, 1e-6)
        cx = _clamp(fx, bx, bx + bw)
        cy = _clamp(fy, by, by + bh)
        mx = self._clamp_x((cx - bx) / bw * self._screen_w)
        my = self._clamp_y((cy - by) / bh * self._screen_h)
        self._last_mapped = (mx, my)

        if self._pending_rebase is not None:
            tx, ty = self._pending_rebase
            self._offset = (tx - mx, ty - my)
            self._pending_rebase = None

        # Internal (never-frozen) position stream; freezing only gates output.
        ix = self._clamp_x(mx + self._offset[0])
        iy = self._clamp_y(my + self._offset[1])

        if self._prev_internal is None:
            self._speed = self._speed_filter.filter(0.0, ts)
        else:
            pt, px, py = self._prev_internal
            dt_s = (ts - pt) / 1000.0
            if dt_s > 0.0:
                raw_speed = math.hypot(ix - px, iy - py) / dt_s
                self._speed = self._speed_filter.filter(raw_speed, ts)
        self._prev_internal = (ts, ix, iy)

        if self._frozen:
            hx, hy = self._last_output
            return CursorSample(hx, hy, self._speed, True, ts)

        ox, oy = ix, iy
        if self._speed < QUANTIZE_MAX_SPEED_PX_S:
            ox, oy = float(round(ox)), float(round(oy))
        self._last_output = (ox, oy)
        return CursorSample(ox, oy, self._speed, False, ts)

    def rebase(self, to_x: float, to_y: float) -> None:
        """Shift the mapping offset so the current output equals the target.

        Used on clutch engage, drag unfreeze, and PALM exit so the cursor
        never jumps. If no frame has been mapped yet, the rebase is applied
        at the first mapped frame instead.
        """
        tx = self._clamp_x(to_x)
        ty = self._clamp_y(to_y)
        if self._last_mapped is None:
            self._pending_rebase = (tx, ty)
        else:
            mx, my = self._last_mapped
            self._offset = (tx - mx, ty - my)
        self._last_output = (tx, ty)

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)

    def set_drag(self, drag: bool) -> None:
        """Switch the position filters between normal and drag smoothing."""
        self._drag = bool(drag)
        oe = self._cfg.one_euro
        mc = oe.drag_mincutoff if self._drag else oe.mincutoff
        self._fx.set_mincutoff(mc)
        self._fy.set_mincutoff(mc)

    def set_beta(self, beta: float) -> None:
        """Live-tune: apply a new beta to the x/y position filters (additive
        to the CONTRACTS.md surface; pinch filters keep their own beta)."""
        self._fx.beta = beta
        self._fy.beta = beta

    def pinch(self, name: str, raw: float, ts_ms: float) -> float:
        """Per-name One Euro (pinch_mincutoff) for pinch distances."""
        f = self._pinch_filters.get(name)
        if f is None:
            oe = self._cfg.one_euro
            f = OneEuro(oe.pinch_mincutoff, oe.beta, oe.dcutoff)
            self._pinch_filters[name] = f
        return f.filter(raw, ts_ms)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._speed_filter.reset()
        self._pinch_filters.clear()
        self._frozen = False
        self._offset = (0.0, 0.0)
        self._pending_rebase = None
        self._last_mapped = None
        self._prev_internal = None
        self._speed = 0.0
        self._last_output = (self._screen_w / 2.0, self._screen_h / 2.0)
        self.set_drag(False)
