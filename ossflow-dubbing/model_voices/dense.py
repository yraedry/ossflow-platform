import subprocess, re
from pathlib import Path

SRC = Path("luis_posada_trim.wav")
OUT = Path("luis_posada_dense.wav")

log = subprocess.run(
    ["ffmpeg", "-hide_banner", "-i", str(SRC),
     "-af", "silencedetect=noise=-30dB:d=0.2", "-f", "null", "-"],
    capture_output=True, text=True,
)
lines = log.stderr.splitlines()

starts, ends = [], []
for ln in lines:
    m = re.search(r"silence_start:\s*([\d.]+)", ln)
    if m: starts.append(float(m.group(1)))
    m = re.search(r"silence_end:\s*([\d.]+)", ln)
    if m: ends.append(float(m.group(1)))

dur = float(subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=nw=1:nk=1", str(SRC)]).decode().strip())

speech = []
prev_end = 0.0
if starts and starts[0] == 0:
    prev_end = ends[0] if ends else 0
    starts = starts[1:]; ends = ends[1:]
for s, e in zip(starts, ends):
    if s - prev_end >= 0.25:
        speech.append((prev_end + 0.02, s - 0.02))
    prev_end = e
if dur - prev_end >= 0.25:
    speech.append((prev_end + 0.02, dur - 0.02))

print(f"Speech segments: {len(speech)}, total={sum(e-s for s,e in speech):.2f}s")

parts = []
for i, (s, e) in enumerate(speech):
    p = Path(f"_part{i:02d}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{s:.3f}", "-i", str(SRC), "-t", f"{e-s:.3f}",
         "-ar", "24000", "-ac", "1", str(p)], check=True)
    parts.append(p)

lst = Path("_concat.txt")
lst.write_text("\n".join(f"file '{p.name}'" for p in parts))

subprocess.run(
    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
     "-f", "concat", "-safe", "0", "-i", str(lst),
     "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-18:TP=-2:LRA=7",
     "-ar", "24000", "-ac", "1", str(OUT)], check=True)

for p in parts: p.unlink()
lst.unlink()

final_dur = float(subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=nw=1:nk=1", str(OUT)]).decode().strip())
print(f"Output: {OUT} ({final_dur:.2f}s)")
