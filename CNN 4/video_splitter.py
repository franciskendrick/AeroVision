import cv2
import os

# ─── User parameters ────────────────────────────────────────────────────────────
videos_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "Videos")
)
INPUT_VIDEO = f"{videos_path}/stop.mp4"
OUTPUT_DIR  = f"{videos_path}/stop" 
BASENAME    = "stop"      # will produce stop_1.mp4, stop_2.mp4, …
# ────────────────────────────────────────────────────────────────────────────────

# create output dir if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# open video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {INPUT_VIDEO!r}")

fps         = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

black_frames = []
idx = 0

print("Scanning for black frames…")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # sample three points:
    pts = [(0, 0), (width-1, height-1), (width//2, height//2)]
    if all((frame[y, x] == [0, 0, 0]).all() for x, y in pts):
        black_frames.append(idx)
    idx += 1

cap.release()

if not black_frames:
    print("No black frames found. Nothing to cut.")
    exit(0)

print(f"Found {len(black_frames)} black frame(s) at indices {black_frames}")

# build segments as (start_frame, end_frame)
segments = []
prev_end = 0
for bf in black_frames:
    # segment runs from prev_end to bf-1
    if bf - 1 >= prev_end:
        segments.append((prev_end, bf - 1))
    prev_end = bf + 1

# final tail segment
if prev_end <= frame_count - 1:
    segments.append((prev_end, frame_count - 1))

print(f"Will write {len(segments)} segment(s): {segments}")

# helper to write a segment
def write_segment(seg_idx, start, end):
    cap_in = cv2.VideoCapture(INPUT_VIDEO)
    cap_in.set(cv2.CAP_PROP_POS_FRAMES, start)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, f"{BASENAME}_{seg_idx+1}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in range(start, end + 1):
        ret, frm = cap_in.read()
        if not ret:
            break
        writer.write(frm)

    writer.release()
    cap_in.release()
    print(f"  • Wrote segment {seg_idx+1}: frames [{start}…{end}] → {out_path}")

# actually write them
for i, (s, e) in enumerate(segments):
    write_segment(i, s, e)

print("Done.")
