import pytest

from athome.execution.screen_record import ScreenRecorder, find_window, monitors, record_command, region_of

XRANDR = """Monitors: 2
 0: +*DP-1 1920/480x1080/260+0+350  DP-1
 1: +HDMI-1-0 1440/597x2560/336+1920+0  HDMI-1-0
"""


def test_monitors_and_regions():
    screens = monitors(XRANDR)
    assert screens == {"DP-1": (1920, 1080, 0, 350), "HDMI-1-0": (1440, 2560, 1920, 0)}
    assert region_of("DP-1", screens) == (1920, 1080, 0, 350)
    assert region_of("all", screens) == (3360, 2560, 0, 0)
    with pytest.raises(ValueError):
        region_of("eDP-1", screens)


def test_command_grabs_the_region_with_the_chosen_encoder():
    cmd = record_command(":0", (1920, 1080, 0, 350), "/tmp/run.mp4", "h264_nvenc")
    assert cmd[cmd.index("-video_size") + 1] == "1920x1080"
    assert cmd[cmd.index("-i") + 1] == ":0+0,350"
    assert "h264_nvenc" in cmd and cmd[-1] == "/tmp/run.mp4"
    assert "libx264" in record_command(":0", (2, 2, 0, 0), "x.mp4", "libx264")


def test_a_recorder_that_cannot_start_reports_it_and_stops_quietly(tmp_path):
    rec = ScreenRecorder("no-such-monitor", display=":0")
    assert rec.start(tmp_path / "a.mp4") is False and rec.error
    assert rec.stop() is None


def test_active_only_while_ffmpeg_runs(tmp_path):
    import subprocess
    rec = ScreenRecorder("DP-1")
    assert rec.active is False
    rec._proc = subprocess.Popen(["sleep", "30"])      # stands in for ffmpeg
    rec._out = tmp_path / "none.mp4"
    assert rec.active is True
    assert rec.stop() is None                          # stopped, no file written
    assert rec.active is False


TREE = """
     0x400006 (has no name): ()  1x1+-1+-1  +-1+-1
     0x4600003 "AtHOME 탐색 콘솔 - Google Chrome": ("google-chrome" "Google-chrome")  1280x900+0+0  +100+37
        0x4600004 "AtHOME 탐색 콘솔 - Google Chrome": ()  1280x860+0+40  +100+77
     0x5200001 "카카오톡": ("kakaotalk" "KakaoTalk")  400x700+0+0  +1500+0
"""


def test_the_console_window_is_found_and_grabbed_by_id():
    assert find_window(TREE) == "0x4600003"
    assert find_window(TREE, "no such title") is None
    cmd = record_command(":0", "0x4600003", "/tmp/run.mp4", "libx264")
    assert cmd[cmd.index("-window_id") + 1] == str(0x4600003) and "-video_size" not in cmd
