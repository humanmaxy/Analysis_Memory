import os
import re
from typing import Dict, List, Tuple


HEADER_RE = re.compile(r"^[0-9a-fA-F]+-[0-9a-fA-F]+\s+\S+\s+\S+\s+\S+\s+\S+(?:\s+(.*))?$")
KV_RE = re.compile(r"^(\w+):\s+(\d+)\s+kB$")


def _classify_path(path: str) -> str:
	path = path or ""
	if not path:
		return "[anon]"
	if path.startswith("[") and path.endswith("]"):
		return path
	if path == "//anon":
		return "[anon]"
	return "file"


def read_smaps(pid: int) -> List[Dict]:
	smaps_path = f"/proc/{pid}/smaps"
	if not os.path.exists(smaps_path):
		raise FileNotFoundError(smaps_path)

	regions: List[Dict] = []
	current: Dict = {
		"path": "",
		"size_kb": 0.0,
		"rss_kb": 0.0,
		"pss_kb": 0.0,
		"private_kb": 0.0,
		"shared_kb": 0.0,
		"swap_kb": 0.0,
		"category": "[anon]",
	}

	def push_current() -> None:
		if current.get("size_kb", 0) > 0:
			regions.append(dict(current))

	with open(smaps_path, "r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			m = HEADER_RE.match(line)
			if m:
				# New region header line begins
				push_current()
				path = (m.group(1) or "").strip()
				current = {
					"path": path,
					"size_kb": 0.0,
					"rss_kb": 0.0,
					"pss_kb": 0.0,
					"private_kb": 0.0,
					"shared_kb": 0.0,
					"swap_kb": 0.0,
					"category": _classify_path(path),
				}
				continue
			km = KV_RE.match(line)
			if km:
				key, val = km.group(1), float(km.group(2))
				if key == "Size":
					current["size_kb"] += val
				elif key == "Rss":
					current["rss_kb"] += val
				elif key == "Pss":
					current["pss_kb"] += val
				elif key == "Private_Clean" or key == "Private_Dirty":
					current["private_kb"] += val
				elif key == "Shared_Clean" or key == "Shared_Dirty":
					current["shared_kb"] += val
				elif key == "Swap":
					current["swap_kb"] += val
		# push last
		push_current()

	return regions


def aggregate_smaps(pid: int, regions: List[Dict] = None) -> Tuple[Dict, Dict[str, Dict]]:
	if regions is None:
		regions = read_smaps(pid)
	totals = {
		"size_kb": 0.0,
		"rss_kb": 0.0,
		"pss_kb": 0.0,
		"private_kb": 0.0,
		"shared_kb": 0.0,
		"swap_kb": 0.0,
	}
	grouped: Dict[str, Dict] = {}

	for r in regions:
		for key in totals:
			totals[key] += float(r.get(key, 0.0))
		cat = r.get("category", "[anon]")
		grp = grouped.setdefault(cat, {
			"size_kb": 0.0,
			"rss_kb": 0.0,
			"pss_kb": 0.0,
			"private_kb": 0.0,
			"shared_kb": 0.0,
			"swap_kb": 0.0,
			"sample_path": r.get("path", ""),
		})
		for key in ("size_kb", "rss_kb", "pss_kb", "private_kb", "shared_kb", "swap_kb"):
			grp[key] += float(r.get(key, 0.0))

	return totals, grouped