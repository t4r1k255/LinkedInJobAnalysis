"""
utils_progress.py
LinkedIn Job Analysis — Progress Bar Utility

Kullanım:
    from utils_progress import ProgressBar, StepTracker

    # Adım bazlı (kaç adım olduğu belli):
    bar = ProgressBar(total=6, title="Cleaning descriptions")
    bar.step("Dollar amounts")      # [██░░░░░░░░]  1/6  Dollar amounts       (2.1s)
    bar.step("Hourly rates")        # [████░░░░░░]  2/6  Hourly rates         (4.8s)
    bar.finish()                    # [██████████]  6/6  Done                 (21.4s)

    # Yüzde bazlı (döngülerde):
    bar = ProgressBar(total=len(df), title="Processing rows", unit="rows")
    for i, row in df.iterrows():
        ...
        bar.update(i + 1)
    bar.finish()

    # Script seviyesi adım takibi:
    tracker = StepTracker(total_steps=7, script_name="model_03_salary_advanced")
    tracker.start(1, "Loading data")
    ...
    tracker.done(1)
    tracker.start(2, "Feature engineering")
"""

import sys
import time
from datetime import datetime


# ── Renkler (terminal destekliyorsa) ─────────────────────────────────────────
class _Color:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[32m"
    CYAN   = "\033[36m"
    YELLOW = "\033[33m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"

    @staticmethod
    def supported():
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    """Renk uygula — terminal desteklemiyorsa düz metin döndür."""
    if _Color.supported():
        return f"{code}{text}{_Color.RESET}"
    return text


# ── Bar karakterleri ──────────────────────────────────────────────────────────
_FILLED  = "█"
_EMPTY   = "░"
_BAR_LEN = 20  # karakter cinsinden toplam bar uzunluğu


def _render_bar(done: int, total: int) -> str:
    """Örn: [████████░░░░░░░░░░░░]"""
    filled = int(_BAR_LEN * done / total) if total > 0 else 0
    bar = _FILLED * filled + _EMPTY * (_BAR_LEN - filled)
    return f"[{bar}]"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0 or done >= total:
        return ""
    rate = done / elapsed
    remaining = (total - done) / rate
    return f"ETA {_fmt_eta_str(remaining)}"


def _fmt_eta_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ── Ana sınıf ─────────────────────────────────────────────────────────────────
class ProgressBar:
    """
    Gerçekçi, adım veya yüzde bazlı progress bar.

    Parametreler
    ------------
    total       : Toplam adım / satır sayısı
    title       : Bar başlığı (sol taraf)
    unit        : Birim etiketi ("steps", "rows", "batches" vb.)
    width       : Bar genişliği (karakter), varsayılan 20
    print_every : update() çağrılarında kaçta bir ekrana yaz (büyük döngüler için)
    """

    def __init__(
        self,
        total: int,
        title: str = "",
        unit: str = "steps",
        width: int = _BAR_LEN,
        print_every: int = 1,
    ):
        self.total       = total
        self.title       = title
        self.unit        = unit
        self.width       = width
        self.print_every = print_every
        self._done       = 0
        self._start      = time.time()
        self._last_label = ""
        self._finished   = False

        if title:
            print(_c(_Color.BOLD, f"\n  {title}"), flush=True)

    # ── İç render ─────────────────────────────────────────────────────────────
    def _print_line(self, label: str = "", final: bool = False):
        elapsed = time.time() - self._start
        filled  = int(self.width * self._done / self.total) if self.total > 0 else self.width
        bar     = _FILLED * filled + _EMPTY * (self.width - filled)

        color = _Color.GREEN if final else _Color.CYAN
        bar_str  = _c(color, f"[{bar}]")
        frac_str = _c(_Color.WHITE, f"{self._done:>{len(str(self.total))}}/{self.total}")
        time_str = _c(_Color.GRAY, f"({_fmt_elapsed(elapsed)})")

        eta = ""
        if not final and self._done > 0 and self._done < self.total:
            eta = _c(_Color.YELLOW, f"  {_fmt_eta(elapsed, self._done, self.total)}")

        label_str = f"  {label}" if label else ""
        line = f"    {bar_str}  {frac_str}  {self.unit}{label_str}  {time_str}{eta}"
        print(line, flush=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def step(self, label: str = ""):
        """Bir adım ilerlet ve ekrana yaz."""
        self._done += 1
        self._last_label = label
        self._print_line(label=label, final=(self._done >= self.total))

    def update(self, current: int, label: str = ""):
        """Mutlak pozisyon ver (döngülerde kullan)."""
        self._done = current
        self._last_label = label
        if current % self.print_every == 0 or current >= self.total:
            self._print_line(label=label, final=(current >= self.total))

    def finish(self, label: str = "Done"):
        """Bar'ı tamamla."""
        if self._finished:
            return
        self._done    = self.total
        self._finished = True
        elapsed = time.time() - self._start
        self._print_line(label=label, final=True)
        print(
            _c(_Color.GRAY, f"    ✓ Completed in {_fmt_elapsed(elapsed)}\n"),
            flush=True,
        )

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


# ── Script seviyesi adım takipçisi ────────────────────────────────────────────
class StepTracker:
    """
    Script genelinde ana adımları zamanlar ve loglar.

    Kullanım:
        tracker = StepTracker(total_steps=7, script_name="model_03")
        tracker.start(1, "Loading data")
        ...  # iş yap
        tracker.done(1)
    """

    def __init__(self, total_steps: int, script_name: str = ""):
        self.total      = total_steps
        self.script     = script_name
        self._step_start: dict[int, float] = {}
        self._t0        = time.time()

        sep = "=" * 82
        print(_c(_Color.BOLD, f"\n{sep}"), flush=True)
        if script_name:
            print(_c(_Color.BOLD, f"  {script_name}"), flush=True)
        print(_c(_Color.BOLD, sep), flush=True)

    def start(self, step: int, label: str):
        self._step_start[step] = time.time()
        ts = datetime.now().strftime("%H:%M:%S")
        print(
            _c(_Color.CYAN, f"\n[{ts}]")
            + _c(_Color.BOLD, f"  Step {step}/{self.total}")
            + f"  {label}",
            flush=True,
        )

    def done(self, step: int, note: str = ""):
        elapsed = time.time() - self._step_start.get(step, time.time())
        suffix  = f"  {note}" if note else ""
        print(
            _c(_Color.GREEN, f"  ✓ Step {step}/{self.total} completed")
            + _c(_Color.GRAY, f" in {_fmt_elapsed(elapsed)}{suffix}"),
            flush=True,
        )

    def finish(self):
        total_elapsed = time.time() - self._t0
        sep = "=" * 82
        print(_c(_Color.BOLD, f"\n{sep}"), flush=True)
        print(
            _c(_Color.GREEN, "  ✓ All steps completed")
            + _c(_Color.GRAY, f" in {_fmt_elapsed(total_elapsed)}"),
            flush=True,
        )
        print(_c(_Color.BOLD, sep), flush=True)


# ── Chunk bazlı yüzde bar (büyük dosyalar için) ───────────────────────────────
class ChunkProgressBar:
    """
    Büyük veri setlerinde chunk'lar üzerinden ilerleme gösterir.
    Her X satırda bir günceller — döngü içinde her satırda print yapmaz.

    Kullanım:
        bar = ChunkProgressBar(total_rows=123_000, chunks=10, title="Processing")
        for i, row in df.iterrows():
            ...
            bar.tick()
        bar.finish()
    """

    def __init__(self, total_rows: int, chunks: int = 10, title: str = ""):
        self.total      = total_rows
        self.chunks     = chunks
        self.chunk_size = max(1, total_rows // chunks)
        self._count     = 0
        self._start     = time.time()
        self._bar       = ProgressBar(
            total=chunks,
            title=title,
            unit="chunks",
            print_every=1,
        )

    def tick(self, n: int = 1):
        self._count += n
        chunk_done = min(self._count // self.chunk_size, self.chunks)
        if chunk_done > self._bar._done:
            rows_k = self._count / 1000
            self._bar.update(
                chunk_done,
                label=f"({rows_k:.1f}k / {self.total/1000:.1f}k rows)",
            )

    def finish(self):
        self._bar.finish(label=f"{self.total:,} rows processed")
