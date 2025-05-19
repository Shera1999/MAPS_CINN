# summarize_missing.py
from collections import Counter

# Read missing filenames
with open("missing_images.txt") as f:
    names = [line.strip() for line in f]

# Parse keys (halo, snapshot)
keys = []
for nm in names:
    base  = nm.replace(".jpg","")
    parts = base.split("_")
    halo, snap = parts[3], parts[1]
    keys.append((halo, snap))

cnt = Counter(keys)
print("Dropped projections per (HaloID, Snapshot):")
for (halo, snap), c in cnt.most_common(20):
    print(f" • Halo {halo}, Snapshot {snap}: {c} projections")

print(f"\nTotal unique clusters dropped: {len(cnt)}")
