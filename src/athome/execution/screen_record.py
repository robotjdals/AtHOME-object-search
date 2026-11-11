"""Screen recording of experiment runs: ffmpeg x11grab of the operator console's
browser window (target "window": only that window, also when other windows
cover it, as the compositor keeps its contents), or of one monitor or all.
GPU encoder when present. A recording that fails never stops the experiment:
start() reports False and the run goes on without a video.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# xrandr --listmonitors: " 0: +*DP-1 1920/480x1080/260+0+350  DP-1"
MONITOR = re.compile(r"^\s*\d+:\s+\+?\*?(\S+)\s+(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)")

Region = Tuple[int, int, int, int]      # width, height, x, y
CONSOLE_TITLE = "AtHOME 탐색 콘솔"        # <title> of athome_ros.operator_console_page
# xwininfo -root -tree: '     0x4600003 "AtHOME 탐색 콘솔 - Google Chrome": ("google-chrome" ...)  1280x900+0+0  +0+37'
WINDOW = re.compile(r'^\s*(0x[0-9a-f]+) "(.*)": .*?\s(\d+)x(\d+)\+-?\d+\+-?\d+')


def find_window(tree_text: str, title: str = CONSOLE_TITLE) -> Optional[str]:
    """The largest window whose name contains ``title`` (browser frames list it more than once)."""
    best = None
    for line in tree_text.splitlines():
        m = WINDOW.match(line)
        if m and title in m.group(2):
            area = int(m.group(3)) * int(m.group(4))
            if best is None or area > best[0]:
                best = (area, m.group(1))
    return None if best is None else best[1]


def monitors(xrandr_text: str) -> Dict[str, Region]:
    found = {}
    for line in xrandr_text.splitlines():
        m = MONITOR.match(line)
        if m:
            name, w, h, x, y = m.groups()
            found[name] = (int(w), int(h), int(x), int(y))
    return found


def region_of(target: str, screens: Dict[str, Region]) -> Region:
    """One monitor by name, or "all": the box around every monitor."""
    if target != "all":
        if target not in screens:
            raise ValueError(f"모니터 {target} 없음 (있는 것: {', '.join(screens) or '없음'})")
        return screens[target]
    x0 = min(x for _, _, x, _ in screens.values())
    y0 = min(y for _, _, _, y in screens.values())
    x1 = max(x + w for w, _, x, _ in screens.values())
    y1 = max(y + h for _, h, _, y in screens.values())
    return x1 - x0, y1 - y0, x0, y0


def record_command(display: str, region, out, encoder: str, fps: int = 15) -> List[str]:
    """``region``: a monitor box (w, h, x, y) or a window id ("0x...")."""
    codec = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28"] if encoder == "h264_nvenc"
             else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"])
    if isinstance(region, str):
        grab = ["-window_id", str(int(region, 16)), "-i", display]
    else:
        w, h, x, y = region
        grab = ["-video_size", f"{w}x{h}", "-i", f"{display}+{x},{y}"]
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "x11grab", "-framerate", str(fps),
            *grab, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", *codec,
            # fragmented MP4: the file plays even if ffmpeg is killed
            "-movflags", "+frag_keyframe+empty_moov", str(out)]


class ScreenRecorder:
    def __init__(self, target: str, display: str = ":0"):
        self.target = target
        self.display = display
        self._proc: Optional[subprocess.Popen] = None
        self._out: Optional[Path] = None
        self.error = ""

    @property
    def active(self) -> bool:
        """ffmpeg is running (shown on the console while a run is recorded)."""
        return self._proc is not None and self._proc.poll() is None

    def _encoder(self) -> str:
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                                  timeout=10).stdout
        return "h264_nvenc" if "h264_nvenc" in encoders else "libx264"

    def start(self, out) -> bool:
        self.stop()
        try:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg 없음")
            if self.target == "window":
                tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True,
                                      timeout=10).stdout
                region = find_window(tree)
                if region is None:
                    raise RuntimeError(f"'{CONSOLE_TITLE}' 창 없음: 이 컴퓨터의 브라우저로 콘솔을 여세요")
            else:
                text = subprocess.run(["xrandr", "--listmonitors"], capture_output=True, text=True,
                                      timeout=10).stdout
                region = region_of(self.target, monitors(text))
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(record_command(self.display, region, out, self._encoder()),
                                          stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._out, self.error = out, ""
            return True
        except Exception as e:  # noqa: BLE001 - the run goes on without a video
            self._proc, self._out, self.error = None, None, f"{type(e).__name__}: {e}"
            return False

    def stop(self) -> Optional[Path]:
        """The file of the recording just stopped, None when none ran."""
        proc, out, self._proc, self._out = self._proc, self._out, None, None
        if proc is None:
            return None
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)      # ffmpeg finishes the file on SIGINT
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        elif proc.returncode not in (0, 255):
            self.error = (proc.stderr.read() or b"").decode(errors="replace").strip()[-300:]
        return out if out is not None and out.is_file() and out.stat().st_size > 0 else None
