#!/usr/bin/env python
"""Download third-party assets listed in ``scripts/assets_manifest.yaml``.

The manifest is the source of truth: per file it records the set, source URL, destination under
``source/kitchenbot/scene/assets/third_party/`` (git-ignored), SHA256, licence and purpose.
``MANIFEST.md`` next to the downloaded files is generated from it (``--write-manifest-md``).

    python scripts/fetch_assets.py --set dishwasher|cabinets|all [--dry-run]

* A file whose SHA256 already matches is skipped, so a second run downloads nothing.
* Hugging Face files go through ``huggingface_hub.hf_hub_download`` (resumable, pinned to the
  manifest revision). MolmoSpaces packs its objects into tar shards, so those entries are fetched
  with an HTTP range request, checked, and unpacked (needs ``zstandard``). Plain URLs
  (``kind: url``) use ``urllib`` with ``.part`` resume.
* ``--record-checksums`` fills in missing SHA256 values after download and rewrites the manifest.
* ``--dry-run`` touches neither the network nor the disk: it reports what a real run would do.

Optional dependencies: ``pip install -e ".[assets]"`` (huggingface_hub, usd-core, zstandard).
Nothing here imports Isaac Sim, Isaac Lab or torch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().with_name("assets_manifest.yaml")
KINDS = ("hf_file", "hf_range", "url")
REQUIRED = ("set", "asset", "source", "kind", "dest", "size", "licence", "purpose")
MANIFEST_HEADER = (
    "# Source of truth for third-party assets, read by scripts/fetch_assets.py.\n"
    "# sha256 of Hugging Face LFS files is the upstream LFS oid; the others were computed at\n"
    "# first download with `fetch_assets.py --record-checksums`. MANIFEST.md is generated from\n"
    "# this file (`fetch_assets.py --write-manifest-md`); do not edit it by hand.\n"
)
CHUNK = 1 << 20


# --------------------------------------------------------------------------- manifest


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    validate_manifest(manifest)
    return manifest


def dump_manifest(manifest: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(MANIFEST_HEADER)
        yaml.safe_dump(manifest, fh, sort_keys=False, width=200, allow_unicode=True)


def validate_manifest(manifest: dict) -> None:
    """Raise ValueError on a malformed manifest (called on every load)."""
    for key in ("dest_root", "sources", "files"):
        if key not in manifest:
            raise ValueError(f"manifest is missing top-level key {key!r}")
    seen: set[str] = set()
    for i, f in enumerate(manifest["files"]):
        where = f"files[{i}] ({f.get('dest', '?')})"
        for key in REQUIRED:
            if key not in f:
                raise ValueError(f"{where}: missing {key!r}")
        if f["kind"] not in KINDS:
            raise ValueError(f"{where}: kind must be one of {KINDS}")
        if f["source"] not in manifest["sources"]:
            raise ValueError(f"{where}: unknown source {f['source']!r}")
        parts = Path(f["dest"]).parts
        if Path(f["dest"]).is_absolute() or ".." in parts:
            raise ValueError(f"{where}: dest must be relative and stay under dest_root")
        if f["dest"] in seen:
            raise ValueError(f"{where}: duplicate dest")
        seen.add(f["dest"])
        if f["kind"] == "hf_file" and f["dest"] != f"{f['source']}/{f['path']}":
            raise ValueError(f"{where}: hf_file dest must be '<source>/<path in repo>'")
        range_keys = ("path", "range_start", "range_length")
        if f["kind"] == "hf_range" and not all(k in f for k in range_keys):
            raise ValueError(f"{where}: hf_range needs path, range_start, range_length")
        if f["kind"] in ("hf_range", "url") and "url" not in f:
            raise ValueError(f"{where}: {f['kind']} needs url")


def set_names(manifest: dict) -> list[str]:
    return sorted({f["set"] for f in manifest["files"]})


def select(manifest: dict, set_name: str) -> list[dict]:
    if set_name == "all":
        return list(manifest["files"])
    return [f for f in manifest["files"] if f["set"] == set_name]


def dest_root(manifest: dict) -> Path:
    return REPO_ROOT / manifest["dest_root"]


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fmt_mb(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


def file_state(entry: dict, root: Path) -> str:
    """One of: ok, missing, mismatch, unrecorded (file present, manifest has no checksum)."""
    path = root / entry["dest"]
    if not path.is_file():
        return "missing"
    if not entry.get("sha256"):
        return "unrecorded"
    return "ok" if sha256_file(path) == entry["sha256"] else "mismatch"


def _import_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("fetch_assets.py needs huggingface_hub: pip install -e '.[assets]'")
    return hf_hub_download


def http_download(url: str, dest: Path, start: int = 0, length: int | None = None) -> None:
    """Fetch ``url`` (optionally bytes [start, start+length)) to ``dest`` via a resumable .part."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if length is not None and have > length:
        part.unlink()
        have = 0
    if length is not None and have == length:
        part.replace(dest)
        return
    first = start + have
    rng = f"bytes={first}-" if length is None else f"bytes={first}-{start + length - 1}"
    headers = {"User-Agent": "kitchenbot-fetch-assets/1"}
    if first > 0 or length is not None:
        headers["Range"] = rng
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        # 200 means the server ignored the Range header: restart from zero.
        mode = "ab" if resp.status == 206 and have else "wb"
        if resp.status == 200 and (start or length is not None):
            raise RuntimeError(f"server ignored the Range header for {url}")
        with open(part, mode) as out:
            shutil.copyfileobj(resp, out, CHUNK)
    part.replace(dest)


def unpack_tar_zst(archive: Path, out_dir: Path, sha256: str) -> None:
    try:
        import zstandard
    except ImportError:
        sys.exit("unpacking MolmoSpaces archives needs zstandard: pip install -e '.[assets]'")
    marker = out_dir / ".extracted"
    if marker.is_file() and marker.read_text().strip() == sha256:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(archive, "rb") as fh:
        raw = zstandard.ZstdDecompressor().stream_reader(fh).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(out_dir, filter="data")
    marker.write_text(sha256 + "\n")


# --------------------------------------------------------------------------- fetch


def fetch_one(entry: dict, manifest: dict, root: Path) -> None:
    """Download ``entry`` to ``root/dest``; raises on failure. Does not verify the checksum."""
    src = manifest["sources"][entry["source"]]
    dest = root / entry["dest"]
    if entry["kind"] == "hf_file":
        hf_hub_download = _import_hf()
        hf_hub_download(
            repo_id=src["repo_id"],
            filename=entry["path"],
            repo_type=src.get("repo_type", "dataset"),
            revision=src.get("revision"),
            local_dir=str(root / entry["source"]),
        )
    elif entry["kind"] == "hf_range":
        http_download(entry["url"], dest, entry["range_start"], entry["range_length"])
    else:
        http_download(entry["url"], dest)


def run(
    manifest: dict, manifest_path: Path, set_name: str, dry_run: bool, record: bool, verbose: bool
) -> int:
    root = dest_root(manifest)
    entries = select(manifest, set_name)
    if not entries:
        print(f"no files in set {set_name!r}; sets: {', '.join(set_names(manifest))}, all")
        return 2
    total = sum(e["size"] for e in entries)
    print(f"set {set_name}: {len(entries)} files, {fmt_mb(total)} -> {root}")

    skipped = downloaded = failed = extracted = 0
    dl_bytes = 0
    changed = False
    for e in entries:
        state = file_state(e, root)
        if state == "ok":
            skipped += 1
            if verbose:
                print(f"  [skip]  {e['dest']}")
        elif state == "unrecorded" and not record:
            skipped += 1
            print(f"  [warn]  {e['dest']}: no sha256 in manifest (use --record-checksums)")
        elif dry_run:
            print(f"  [would] {state:8s} {e['dest']} ({fmt_mb(e['size'])})")
            downloaded += 1
            dl_bytes += e["size"]
        else:
            try:
                if state in ("missing", "mismatch"):
                    if state == "mismatch":
                        (root / e["dest"]).unlink()
                    fetch_one(e, manifest, root)
                    downloaded += 1
                    dl_bytes += e["size"]
                    print(f"  [get]   {e['dest']} ({fmt_mb(e['size'])})")
                digest = sha256_file(root / e["dest"])
                if not e.get("sha256"):
                    e["sha256"] = digest
                    changed = True
                elif digest != e["sha256"]:
                    (root / e["dest"]).unlink()
                    raise RuntimeError(f"sha256 mismatch: got {digest}, manifest {e['sha256']}")
            except Exception as exc:  # report and continue with the other files
                failed += 1
                print(f"  [FAIL]  {e['dest']}: {exc}")
                continue
        if not dry_run and e.get("extract_to") and (root / e["dest"]).is_file():
            before = (root / e["extract_to"] / ".extracted").is_file()
            unpack_tar_zst(root / e["dest"], root / e["extract_to"], e["sha256"])
            if not before:
                extracted += 1
                print(f"  [unpack] {e['extract_to']}")

    if changed and record:
        dump_manifest(manifest, manifest_path)
        print(f"recorded missing checksums in {manifest_path}")
    verb = "would download" if dry_run else "downloaded"
    print(
        f"summary: {verb} {downloaded} files ({fmt_mb(dl_bytes)}), skipped {skipped} "
        f"(checksum matched), unpacked {extracted}, failed {failed}"
    )
    return 1 if failed else 0


# --------------------------------------------------------------------------- MANIFEST.md


def render_manifest_md(manifest: dict) -> str:
    out = [
        "# Third-party assets",
        "",
        "Generated from `scripts/assets_manifest.yaml` by `python scripts/fetch_assets.py "
        "--write-manifest-md`; do not edit by hand. The manifest and this file are committed, the "
        "files themselves are git-ignored and fetched with `python scripts/fetch_assets.py "
        "--set dishwasher|cabinets|all`.",
        "",
        "## Sources",
        "",
        "| Source | URL | Pinned revision | Licence | Reference |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, s in manifest["sources"].items():
        out.append(
            f"| {s['name']} (`{key}`) | <{s['url']}> | `{s.get('revision', '-')}` | "
            f"{s['licence']} | {s.get('paper', '-')} |"
        )
    out += [
        "",
        "Licence notes. ArtVIP: the Hugging Face dataset card metadata says `apache-2.0`; the "
        "repo ships no separate LICENSE file. MolmoSpaces: the card says all subsets except "
        "Objaverse are CC BY 4.0 (attribution required: Ai2, MolmoSpaces, arXiv 2602.11337) and "
        "adds that the data is intended for research and educational use under Ai2's "
        "Responsible Use Guidelines. The `commercial_safe` filter exists only in the code "
        "(<https://github.com/allenai/molmospaces>, an environment variable that drops "
        "non-commercial Objaverse assets); the dataset card has no such flag and the THOR "
        "assets used here are not Objaverse.",
        "",
        "## Assets",
        "",
        "| Set | Asset | Source | Files | Size | Licence | What we use it for |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in manifest["files"]:
        groups.setdefault((f["set"], f["asset"]), []).append(f)
    for (set_name, asset), fs in groups.items():
        size = sum(f["size"] for f in fs)
        out.append(
            f"| {set_name} | `{asset}` | {fs[0]['source']} | {len(fs)} | {fmt_mb(size)} | "
            f"{fs[0]['licence']} | {fs[0]['purpose']} |"
        )
    total = sum(f["size"] for f in manifest["files"])
    out += [
        "",
        f"Total: {len(manifest['files'])} downloaded files, {fmt_mb(total)} "
        "(cap 2 GB). MolmoSpaces packs objects into tar shards: each `.tar.zst` is fetched by "
        "HTTP byte range from the shard named in the YAML and unpacked next to the archive.",
        "",
        "## Files",
        "",
        "Paths are relative to `source/kitchenbot/scene/assets/third_party/`. `sha256` is the "
        "first 16 hex digits; the full value is in the YAML.",
        "",
        "| Set | File | Size (bytes) | sha256 | Licence |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for f in manifest["files"]:
        digest = (f.get("sha256") or "unrecorded")[:16]
        out.append(f"| {f['set']} | `{f['dest']}` | {f['size']} | `{digest}` | {f['licence']} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", dest="set_name", help="dishwasher, cabinets or all (see manifest)")
    ap.add_argument("--dry-run", action="store_true", help="report only; no network, no writes")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest YAML")
    ap.add_argument(
        "--record-checksums", action="store_true", help="fill missing sha256 values and save"
    )
    ap.add_argument(
        "--write-manifest-md", action="store_true", help="regenerate third_party/MANIFEST.md"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="also list skipped files")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)

    if args.write_manifest_md:
        target = dest_root(manifest) / "MANIFEST.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_manifest_md(manifest), encoding="utf-8", newline="\n")
        print(f"wrote {target}")
        if not args.set_name:
            return 0
    if not args.set_name:
        ap.error("--set is required (dishwasher, cabinets or all)")
    return run(
        manifest, args.manifest, args.set_name, args.dry_run, args.record_checksums, args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
