from __future__ import annotations

import argparse
from pathlib import Path

from .core import scan_library, make_contact_sheets, organize, diagnose_folders


def main():
    parser = argparse.ArgumentParser(prog="ndl_library_manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan SFW/NSFW folders and create CSV indexes")
    scan.add_argument("--sfw", type=Path, default=None)
    scan.add_argument("--nsfw", type=Path, default=None)
    scan.add_argument("--out", type=Path, required=True)
    scan.add_argument("--gap-hours", type=float, default=4.0)
    scan.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    scan.add_argument("--duplicate-mode", choices=["fast", "full"], default="fast", help="fast uses indexed candidate search; full compares every pair")
    scan.add_argument("--duplicate-threshold", type=int, default=6, help="perceptual hash distance for duplicate grouping")

    diag = sub.add_parser("diagnose", help="Check input folders and count supported images")
    diag.add_argument("--sfw", type=Path, default=None)
    diag.add_argument("--nsfw", type=Path, default=None)

    sheets = sub.add_parser("contact-sheets", help="Create review contact sheets from scan output")
    sheets.add_argument("--out", type=Path, required=True)
    sheets.add_argument("--max-per-sheet", type=int, default=40)

    org = sub.add_parser("organize", help="Copy images into an organized Patreon-ready structure")
    org.add_argument("--out", type=Path, required=True)
    org.add_argument("--destination", type=Path, required=True)
    org.add_argument("--free-ratio", type=float, default=0.10)

    args = parser.parse_args()
    if args.cmd == "scan":
        scan_library(args.sfw, args.nsfw, args.out, gap_hours=args.gap_hours, verbose=not args.quiet, duplicate_mode=args.duplicate_mode, duplicate_threshold=args.duplicate_threshold)
    elif args.cmd == "diagnose":
        diagnose_folders(args.sfw, args.nsfw)
    elif args.cmd == "contact-sheets":
        make_contact_sheets(args.out, max_per_sheet=args.max_per_sheet)
    elif args.cmd == "organize":
        organize(args.out, args.destination, free_ratio=args.free_ratio)

if __name__ == "__main__":
    main()
