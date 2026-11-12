"""Video clips of the project page (docs/assets/clips/), re-encoded for the web:

    overview.mp4 / overview.jpg   the cut-edited four-case video, H.264, no sound, one still
    task.mp4 / task.jpg           one whole search task of the final experiment (optional, --task)
    sNN_trained.mp4, sNN_base.mp4 the operator-console recording of a placement, cropped to the
                                  status card and the camera view, at 8x with a real-time clock
    sNN_trained.jpg, sNN_base.jpg the first frame of each clip, shown before it plays

    python3 scripts/build_page_clips.py --overview ~/Downloads/<edited video>.mp4
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURED = {17: (34, 35), 13: (26, 27), 4: (8, 7), 15: (30, 31)}   # placement -> run numbers (trained, base)
# The status card and the camera view of the operator console, by recording size. The 2740x1566
# recordings (placements 15 and 17) were made in a browser window: the crop starts below its tab,
# address and bookmark bars. The right column (run table, paths) is outside both.
CROPS = {2880: "crop=1988:1438:24:158", 2740: "crop=1850:1142:24:400"}
SPEED = 8
LEAD_IN = 0.5                                        # s: the first frames still show the previous run's result
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
H264 = ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def run_clip(src, out):
    width = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width", "-of", "csv=p=0", src]))
    clock = (f"drawtext=fontfile={FONT}:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=12:x=w-tw-28:y=28:"
             f"text='%{{eif\\:floor(t*{SPEED}/60)\\:d\\:2}}\\:%{{eif\\:mod(floor(t*{SPEED})\\,60)\\:d\\:2}}  {SPEED}x'")
    ffmpeg("-ss", LEAD_IN, "-i", src, "-vf", f"{CROPS[width]},setpts=PTS/{SPEED},fps=15,scale=800:-2,{clock}", *H264, "-crf", "29", out)
    ffmpeg("-ss", "0.2", "-i", out, "-frames:v", 1, "-q:v", 5, out.with_suffix(".jpg"))   # the graph starts at "Ready" too


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overview", type=Path, required=True)
    ap.add_argument("--task", type=Path)
    ap.add_argument("--runs", type=Path, default=ROOT / "outputs/room_baram/experiments/room_baram_final/videos")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/assets/clips")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ffmpeg("-i", args.overview, "-vf", "scale=1280:-2", *H264, "-crf", "26", args.output / "overview.mp4")
    ffmpeg("-ss", "29.5", "-i", args.overview, "-frames:v", 1, "-vf", "scale=1280:-2", "-q:v", 4, args.output / "overview.jpg")
    if args.task:
        ffmpeg("-i", args.task, "-vf", "scale=1280:-2", *H264, "-crf", "26", args.output / "task.mp4")
        ffmpeg("-ss", "40", "-i", args.task, "-frames:v", 1, "-vf", "scale=1280:-2", "-q:v", 4, args.output / "task.jpg")
    for scenario, (trained, base) in FEATURED.items():
        for who, number in (("trained", trained), ("base", base)):
            (src,) = args.runs.glob(f"run_{number:02d}_s{scenario:02d}_{who}.mp4")
            run_clip(src, args.output / f"s{scenario}_{who}.mp4")


if __name__ == "__main__":
    main()
