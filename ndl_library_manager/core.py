from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ExifTags

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

EXIF_DATE_TAGS = {
    "DateTimeOriginal",
    "DateTimeDigitized",
    "DateTime",
}

@dataclass
class ImageRecord:
    id: str
    collection: str
    path: str
    filename: str
    created_at: str
    created_source: str
    width: int
    height: int
    aspect: float
    phash: str
    ahash: str
    dhash: str
    mean_r: float
    mean_g: float
    mean_b: float
    brightness: float
    session_id: str = ""
    duplicate_group: str = ""
    suggested_bucket: str = ""


def iter_images(folder: Path) -> Iterable[Path]:
    for root, _, files in os.walk(folder):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in IMAGE_EXTS:
                yield p


def _parse_exif_dt(value: str) -> Optional[datetime]:
    # EXIF often uses YYYY:MM:DD HH:MM:SS
    value = str(value).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            pass
    return None


def read_created_at(path: Path) -> tuple[datetime, str]:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                tag_by_id = {v: k for k, v in ExifTags.TAGS.items()}
                for tag_name in EXIF_DATE_TAGS:
                    tag_id = tag_by_id.get(tag_name)
                    if tag_id and tag_id in exif:
                        dt = _parse_exif_dt(exif.get(tag_id))
                        if dt:
                            return dt, f"exif:{tag_name}"
    except Exception:
        pass

    # Google Takeout sidecar support, common variants
    for sidecar in (Path(str(path) + ".json"), path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                ts = data.get("photoTakenTime", {}).get("timestamp") or data.get("creationTime", {}).get("timestamp")
                if ts:
                    return datetime.fromtimestamp(int(ts)), f"sidecar:{sidecar.name}"
            except Exception:
                pass

    return datetime.fromtimestamp(path.stat().st_mtime), "file_modified"


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:80]


def scan_image(path: Path, collection: str, root: Path) -> Optional[ImageRecord]:
    try:
        created_at, source = read_created_at(path)
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            rgb = img.convert("RGB")
            small = rgb.resize((64, 64))
            arr = np.asarray(small).astype(np.float32)
            mean = arr.mean(axis=(0, 1))
            brightness = float(arr.mean())
            ph = imagehash.phash(rgb)
            ah = imagehash.average_hash(rgb)
            dh = imagehash.dhash(rgb)
        rel = path.relative_to(root) if path.is_relative_to(root) else path.name
        ident = _safe_id(f"{collection}_{created_at.strftime('%Y%m%d_%H%M%S')}_{path.stem}")
        return ImageRecord(
            id=ident,
            collection=collection,
            path=str(path),
            filename=str(rel),
            created_at=created_at.isoformat(sep=" "),
            created_source=source,
            width=width,
            height=height,
            aspect=round(width / height, 4) if height else 0,
            phash=str(ph),
            ahash=str(ah),
            dhash=str(dh),
            mean_r=float(mean[0]),
            mean_g=float(mean[1]),
            mean_b=float(mean[2]),
            brightness=brightness,
        )
    except Exception as e:
        print(f"Skipped {path}: {e}")
        return None


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def color_distance(row_a, row_b) -> float:
    return math.sqrt(
        (row_a.mean_r - row_b.mean_r) ** 2 +
        (row_a.mean_g - row_b.mean_g) ** 2 +
        (row_a.mean_b - row_b.mean_b) ** 2
    )


def assign_sessions(df: pd.DataFrame, gap_hours: float = 4.0, merge_hash_distance: int = 10) -> pd.DataFrame:
    df = df.sort_values(["collection", "created_at", "filename"]).reset_index(drop=True)
    session_counter = 0
    session_ids = []
    prev = None
    for row in df.itertuples():
        dt = pd.to_datetime(row.created_at, format="mixed", errors="coerce")
        new_session = prev is None or row.collection != prev.collection
        if prev is not None and row.collection == prev.collection:
            prev_dt = pd.to_datetime(prev.created_at, format="mixed", errors="coerce")
            gap = (dt - prev_dt).total_seconds() / 3600
            hash_close = hamming_hex(row.phash, prev.phash) <= merge_hash_distance
            color_close = color_distance(row, prev) <= 35
            if gap > gap_hours and not (gap <= 30 and hash_close and color_close):
                new_session = True
        if new_session:
            session_counter += 1
        session_ids.append(f"S{session_counter:04d}")
        prev = row
    df["session_id"] = session_ids
    return df


def _hash_prefix_buckets(df: pd.DataFrame, prefix_chars: int = 3) -> dict[str, list[int]]:
    """Small candidate buckets for fast duplicate search.

    Perceptual hashes that are very close do not always share a prefix, so this is
    only one candidate source. Session/time candidates are still used too.
    """
    buckets: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        for key in (str(row.phash)[:prefix_chars], str(row.ahash)[:prefix_chars], str(row.dhash)[:prefix_chars]):
            buckets.setdefault(f"{row.collection}:{key}", []).append(idx)
    return buckets


def _union_find(n: int):
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return parent, find, union


def assign_duplicate_groups(
    df: pd.DataFrame,
    threshold: int = 6,
    mode: str = "fast",
    session_window: int = 1,
    max_time_gap_hours: float = 36.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assign duplicate groups using perceptual-hash Hamming distance.

    fast mode avoids all-vs-all comparison. It compares likely candidates only:
    - same collection
    - same session or neighboring sessions
    - files within a configurable time window
    - similar aspect ratio
    - small hash-prefix buckets

    full mode keeps the old all-vs-all behavior for small libraries/testing.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    groups = [""] * n
    if n <= 1:
        df["duplicate_group"] = groups
        return df

    dt = pd.to_datetime(df.created_at, format="mixed", errors="coerce")
    session_nums = df.session_id.astype(str).str.extract(r"(\d+)").fillna(-999999).astype(int)[0].to_numpy()
    aspects = df.aspect.astype(float).to_numpy()
    parent, find, union = _union_find(n)
    candidate_pairs: set[tuple[int, int]] = set()

    def maybe_add_pair(i: int, j: int) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        candidate_pairs.add((i, j))

    if mode == "full":
        total = n * (n - 1) // 2
        if verbose:
            print(f"Duplicate mode: full brute force ({total:,} comparisons)")
        for i in range(n):
            for j in range(i + 1, n):
                maybe_add_pair(i, j)
    else:
        if verbose:
            print("Duplicate mode: fast indexed search")

        # Candidate source 1: same or neighboring sessions, but only nearby time/aspect.
        by_collection = {c: list(g.index) for c, g in df.groupby("collection", sort=False)}
        for collection, indices in by_collection.items():
            if verbose:
                print(f"  building candidates for {collection}: {len(indices)} images")
            indices = sorted(indices, key=lambda i: dt.iloc[i])
            for pos, i in enumerate(indices):
                # Walk forward until outside max_time_gap_hours. Usually this is tiny.
                for j in indices[pos + 1:]:
                    hours = abs((dt.iloc[j] - dt.iloc[i]).total_seconds()) / 3600.0
                    if hours > max_time_gap_hours and abs(session_nums[j] - session_nums[i]) > session_window:
                        # Sorted by time, so later j only get farther away.
                        break
                    if abs(session_nums[j] - session_nums[i]) <= session_window or hours <= max_time_gap_hours:
                        if abs(aspects[i] - aspects[j]) <= 0.08:
                            maybe_add_pair(i, j)

        # Candidate source 2: shared hash prefix buckets, catches duplicate echoes across sessions.
        for bucket, idxs in _hash_prefix_buckets(df, prefix_chars=3).items():
            if len(idxs) < 2:
                continue
            if len(idxs) > 500:
                # Very broad bucket. Skip to avoid accidental swamp.
                continue
            for a_pos, i in enumerate(idxs):
                for j in idxs[a_pos + 1:]:
                    if df.at[i, "collection"] == df.at[j, "collection"] and abs(aspects[i] - aspects[j]) <= 0.08:
                        maybe_add_pair(i, j)

        if verbose:
            old_total = n * (n - 1) // 2
            print(f"  candidate comparisons: {len(candidate_pairs):,} instead of {old_total:,}")

    checked = 0
    hits = 0
    for i, j in sorted(candidate_pairs):
        checked += 1
        if verbose and (checked == 1 or checked % 50000 == 0 or checked == len(candidate_pairs)):
            print(f"  duplicate compare {checked:,}/{len(candidate_pairs):,}")
        row_i = df.iloc[i]
        row_j = df.iloc[j]
        if row_i.collection != row_j.collection:
            continue
        # Combine pHash + dHash; pHash catches semantic similarity, dHash catches structural duplication.
        ph = hamming_hex(row_i.phash, row_j.phash)
        dh = hamming_hex(row_i.dhash, row_j.dhash)
        ah = hamming_hex(row_i.ahash, row_j.ahash)
        if ph <= threshold or (ph <= threshold + 2 and dh <= threshold + 2) or (dh <= threshold and ah <= threshold + 2):
            union(i, j)
            hits += 1

    roots: dict[int, list[int]] = {}
    for i in range(n):
        roots.setdefault(find(i), []).append(i)

    group_num = 0
    for members in roots.values():
        if len(members) > 1:
            group_num += 1
            gid = f"D{group_num:04d}"
            for idx in members:
                groups[idx] = gid

    if verbose:
        print(f"  duplicate links found: {hits:,}")
        print(f"  duplicate groups assigned: {group_num:,}")

    df["duplicate_group"] = groups
    return df

def make_sessions_csv(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for sid, g in df.groupby("session_id"):
        start = pd.to_datetime(g.created_at, format="mixed", errors="coerce").min()
        end = pd.to_datetime(g.created_at, format="mixed", errors="coerce").max()
        collection = g.collection.iloc[0]
        rows.append({
            "session_id": sid,
            "collection": collection,
            "start": start,
            "end": end,
            "count": len(g),
            "rough_theme_label": "",
            "notes": "",
        })
    pd.DataFrame(rows).sort_values(["collection", "start"]).to_csv(out_dir / "sessions.csv", index=False)


def make_duplicate_csv(df: pd.DataFrame, out_dir: Path) -> None:
    dups = df[df.duplicate_group != ""].sort_values(["duplicate_group", "created_at"])
    dups[["duplicate_group", "collection", "session_id", "created_at", "filename", "path"]].to_csv(out_dir / "duplicate_groups.csv", index=False)


def diagnose_folders(sfw: Optional[Path], nsfw: Optional[Path]) -> None:
    """Print a clear image/file count before running a scan."""
    print("NDL diagnose")
    print("============")
    total_images = 0
    for collection, folder in (("SFW", sfw), ("NSFW", nsfw)):
        if not folder:
            print(f"{collection}: not supplied")
            continue
        folder = Path(folder)
        print(f"\n{collection}: {folder}")
        print(f"exists: {folder.exists()} | is_dir: {folder.is_dir()}")
        if not folder.exists():
            continue
        all_files = [p for p in folder.rglob("*") if p.is_file()]
        images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS]
        total_images += len(images)
        print(f"files: {len(all_files)} | supported images: {len(images)}")
        suffixes = {}
        for p in all_files:
            suffixes[p.suffix.lower() or "<none>"] = suffixes.get(p.suffix.lower() or "<none>", 0) + 1
        print("extensions:", ", ".join(f"{k}:{v}" for k, v in sorted(suffixes.items())))
        for p in images[:5]:
            print(f"sample: {p}")
    print(f"\nTotal supported images: {total_images}")


def scan_library(
    sfw: Optional[Path],
    nsfw: Optional[Path],
    out_dir: Path,
    gap_hours: float = 4.0,
    verbose: bool = True,
    duplicate_mode: str = "fast",
    duplicate_threshold: int = 6,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ImageRecord] = []
    print(f"Output folder: {out_dir}")
    for collection, folder in (("SFW", sfw), ("NSFW", nsfw)):
        if not folder:
            print(f"{collection}: skipped, no folder supplied")
            continue
        folder = Path(folder)
        print(f"{collection}: scanning {folder}")
        if not folder.exists() or not folder.is_dir():
            print(f"{collection}: folder not found or not a directory")
            continue
        images = list(iter_images(folder))
        print(f"{collection}: found {len(images)} supported images")
        for idx, img_path in enumerate(images, start=1):
            if verbose and (idx == 1 or idx % 25 == 0 or idx == len(images)):
                print(f"{collection}: hashing {idx}/{len(images)} - {img_path.name}")
            rec = scan_image(img_path, collection, folder)
            if rec:
                records.append(rec)
    if not records:
        print("No images were scanned. Check paths and supported extensions.")
        raise SystemExit(1)
    print(f"Assigning sessions for {len(records)} images...")
    df = pd.DataFrame([asdict(r) for r in records])
    df = assign_sessions(df, gap_hours=gap_hours)
    print("Finding near duplicates...")
    df = assign_duplicate_groups(df, threshold=duplicate_threshold, mode=duplicate_mode, verbose=verbose)
    manifest = out_dir / "library_manifest.csv"
    df.to_csv(manifest, index=False)
    make_sessions_csv(df, out_dir)
    make_duplicate_csv(df, out_dir)
    print(f"Scanned {len(df)} images into {out_dir}")
    print(f"Wrote: {manifest}")
    print(f"Wrote: {out_dir / 'sessions.csv'}")
    print(f"Wrote: {out_dir / 'duplicate_groups.csv'}")


def contact_sheet_for_group(group: pd.DataFrame, dest: Path, title: str, thumb_size=(220, 220), cols: int = 5):
    rows = math.ceil(len(group) / cols)
    header_h = 42
    cell_w, cell_h = thumb_size[0], thumb_size[1] + 58
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black")
    for idx, row in enumerate(group.itertuples()):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h + header_h
        try:
            with Image.open(row.path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail(thumb_size)
                tx = x + (thumb_size[0] - img.width) // 2
                ty = y
                sheet.paste(img, (tx, ty))
        except Exception:
            pass
        label = f"{row.session_id} | {Path(row.filename).name[:24]}"
        date = str(row.created_at)[:16]
        draw.text((x + 4, y + thumb_size[1] + 4), label, fill="black")
        draw.text((x + 4, y + thumb_size[1] + 22), date, fill="black")
        if row.duplicate_group:
            draw.text((x + 4, y + thumb_size[1] + 40), row.duplicate_group, fill="black")
    sheet.save(dest, quality=88)


def make_contact_sheets(out_dir: Path, max_per_sheet: int = 40) -> None:
    df = pd.read_csv(out_dir / "library_manifest.csv")
    sheets_dir = out_dir / "contact_sheets"
    sheets_dir.mkdir(exist_ok=True)
    for sid, g in df.groupby("session_id"):
        g = g.sort_values("created_at")
        chunks = [g.iloc[i:i+max_per_sheet] for i in range(0, len(g), max_per_sheet)]
        for n, chunk in enumerate(chunks, start=1):
            dest = sheets_dir / f"{sid}_{n:02d}.jpg"
            title = f"{sid} | {chunk.collection.iloc[0]} | {len(chunk)} images"
            contact_sheet_for_group(chunk, dest, title)
    print(f"Contact sheets written to {sheets_dir}")


def _copy_unique(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
        return
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while True:
        candidate = dest.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return
        i += 1


def organize(out_dir: Path, destination: Path, free_ratio: float = 0.10) -> None:
    df = pd.read_csv(out_dir / "library_manifest.csv")
    df["dt"] = pd.to_datetime(df.created_at, format="mixed", errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)
    cutoff_index = max(1, int(len(df) * free_ratio))
    free_ids = set(df.iloc[:cutoff_index].id)

    sessions_path = out_dir / "sessions.csv"
    session_labels = {}
    if sessions_path.exists():
        s = pd.read_csv(sessions_path).fillna("")
        session_labels = dict(zip(s.session_id, s.rough_theme_label))

    for row in df.itertuples():
        tier = "Patreon_Free_Oldest_10pct" if row.id in free_ids else "Paid_Member_Library"
        label = session_labels.get(row.session_id, "") or row.session_id
        label = _safe_id(label)
        dest = destination / row.collection / tier / label / Path(row.filename).name
        _copy_unique(Path(row.path), dest)
    print(f"Organized copy written to {destination}")
