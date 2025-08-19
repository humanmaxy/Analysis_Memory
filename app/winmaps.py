import os
from typing import Dict, List, Tuple

import psutil


def read_win_maps(pid: int) -> List[Dict]:
	proc = psutil.Process(pid)
	regions: List[Dict] = []
	try:
		for m in proc.memory_maps(grouped=True):
			path = m.path or "[anon]"
			base = os.path.basename(path) if path and not path.startswith("[") else path
			regions.append({
				"path": path,
				"size_kb": 0.0,
				"rss_kb": float(getattr(m, "rss", 0)) / 1024.0,
				"pss_kb": 0.0,
				"private_kb": float(getattr(m, "private", 0)) / 1024.0,
				"shared_kb": float(getattr(m, "shared", 0)) / 1024.0,
				"swap_kb": 0.0,
				"category": base or "[anon]",
			})
	except psutil.AccessDenied:
		raise
	except psutil.NoSuchProcess:
		raise
	except Exception:
		# Fallback: nothing
		pass
	return regions


def aggregate_win_maps(pid: int, regions: List[Dict] = None) -> Tuple[Dict, Dict[str, Dict]]:
	if regions is None:
		regions = read_win_maps(pid)
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