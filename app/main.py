import os
import sys
import time
import threading
import shutil
import subprocess
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import psutil

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
	QApplication,
	QCheckBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QMessageBox,
	QPushButton,
	QPlainTextEdit,
	QSlider,
	QSpinBox,
	QSplitter,
	QTableWidget,
	QTableWidgetItem,
	QTabWidget,
	QVBoxLayout,
	QWidget,
	QFileDialog,
)

import pyqtgraph as pg

from .smaps import read_smaps, aggregate_smaps


@dataclass
class MemorySample:
	timestamp: float
	rss: int
	vms: int
	uss: int
	pss: int
	swap: int
	cpu_percent: float
	threads: int


class ProcessMonitor(QThread):
	sampled = pyqtSignal(object)  # MemorySample
	errored = pyqtSignal(str)
	exited = pyqtSignal()

	def __init__(self, pid: int, interval_seconds: float = 1.0) -> None:
		super().__init__()
		self.pid = pid
		self.interval_seconds = max(0.2, float(interval_seconds))
		self._stop_event = threading.Event()
		self._process: Optional[psutil.Process] = None
		self._initialized_cpu = False

	def stop(self) -> None:
		self._stop_event.set()

	def run(self) -> None:
		try:
			self._process = psutil.Process(self.pid)
		except psutil.NoSuchProcess:
			self.errored.emit(f"进程 {self.pid} 不存在")
			return
		except Exception as exc:
			self.errored.emit(f"无法附加到进程 {self.pid}: {exc}")
			return

		# Warm-up CPU percent to avoid 0.0 values
		try:
			self._process.cpu_percent(interval=None)
			self._initialized_cpu = True
		except Exception:
			pass

		while not self._stop_event.is_set():
			try:
				if self._process is None:
					break
				mem = self._safe_memory_full_info(self._process)
				cpu_percent = 0.0
				try:
					cpu_percent = self._process.cpu_percent(interval=None)
				except Exception:
					cpu_percent = 0.0
				threads = 0
				try:
					threads = self._process.num_threads()
				except Exception:
					threads = 0

				sample = MemorySample(
					timestamp=time.time(),
					rss=mem.get("rss", 0),
					vms=mem.get("vms", 0),
					uss=mem.get("uss", 0),
					pss=mem.get("pss", 0),
					swap=mem.get("swap", 0),
					cpu_percent=cpu_percent,
					threads=threads,
				)
				self.sampled.emit(sample)
			except psutil.NoSuchProcess:
				self.exited.emit()
				break
			except Exception as exc:
				self.errored.emit(str(exc))
				break

			self._stop_event.wait(self.interval_seconds)

	@staticmethod
	def _safe_memory_full_info(process: psutil.Process) -> Dict[str, int]:
		info: Dict[str, int] = {"rss": 0, "vms": 0, "uss": 0, "pss": 0, "swap": 0}
		try:
			basic = process.memory_info()
			info["rss"] = getattr(basic, "rss", 0) or 0
			info["vms"] = getattr(basic, "vms", 0) or 0
		except Exception:
			pass
		try:
			full = process.memory_full_info()
			for key in ("uss", "pss", "swap"):
				value = getattr(full, key, 0)
				if isinstance(value, (int, float)):
					info[key] = int(value)
		except Exception:
			# Fallback using smaps for PSS if accessible
			try:
				totals, _ = aggregate_smaps(process.pid)
				info["pss"] = int(totals.get("pss_kb", 0) * 1024)
				info["uss"] = int(totals.get("private_kb", 0) * 1024)
				info["swap"] = int(totals.get("swap_kb", 0) * 1024)
			except Exception:
				pass
		return info


class OverviewTab(QWidget):
	snapshotTaken = pyqtSignal(object)  # MemorySample

	def __init__(self) -> None:
		super().__init__()
		self.pid: Optional[int] = None
		self.monitor: Optional[ProcessMonitor] = None
		self.history: Deque[MemorySample] = deque(maxlen=600)

		self._build_ui()

	def _build_ui(self) -> None:
		layout = QVBoxLayout(self)

		self.summary_label = QLabel("未选择进程")
		layout.addWidget(self.summary_label)

		# Graphs
		self.plot = pg.PlotWidget()
		self.plot.setBackground("w")
		self.plot.addLegend()
		self.plot.showGrid(x=True, y=True, alpha=0.3)
		self.rss_curve = self.plot.plot([], [], pen=pg.mkPen(color=(200, 0, 0), width=2), name="RSS (MB)")
		self.pss_curve = self.plot.plot([], [], pen=pg.mkPen(color=(0, 120, 200), width=2), name="PSS (MB)")
		layout.addWidget(self.plot, stretch=1)

		controls = QHBoxLayout()
		controls.addWidget(QLabel("采样间隔(毫秒):"))
		self.interval_slider = QSlider(Qt.Horizontal)
		self.interval_slider.setRange(100, 5000)
		self.interval_slider.setValue(1000)
		controls.addWidget(self.interval_slider)

		self.start_btn = QPushButton("开始监控")
		self.stop_btn = QPushButton("停止")
		self.snapshot_btn = QPushButton("拍摄快照")
		controls.addWidget(self.start_btn)
		controls.addWidget(self.stop_btn)
		controls.addWidget(self.snapshot_btn)
		controls.addStretch(1)

		layout.addLayout(controls)

		self.start_btn.clicked.connect(self._on_start)
		self.stop_btn.clicked.connect(self._on_stop)
		self.snapshot_btn.clicked.connect(self._on_snapshot)
		self.interval_slider.valueChanged.connect(self._on_interval_changed)

	def attach_to_pid(self, pid: int) -> None:
		if self.monitor:
			self.monitor.stop()
			self.monitor.wait(1000)
			self.monitor = None
			self.history.clear()
			self._update_plot()
		self.pid = pid
		self.summary_label.setText(f"已选择进程 PID={pid}")

	def _on_interval_changed(self, value_ms: int) -> None:
		if self.monitor:
			self.monitor.interval_seconds = max(0.2, value_ms / 1000.0)

	def _on_start(self) -> None:
		if not self.pid:
			QMessageBox.warning(self, "提示", "请先在左侧选择一个进程")
			return
		if self.monitor:
			self.monitor.stop()
			self.monitor.wait(500)
		self.monitor = ProcessMonitor(self.pid, self.interval_slider.value() / 1000.0)
		self.monitor.sampled.connect(self._on_sampled)
		self.monitor.errored.connect(lambda msg: QMessageBox.critical(self, "错误", msg))
		self.monitor.exited.connect(lambda: QMessageBox.information(self, "提示", "进程已退出"))
		self.monitor.start()

	def _on_stop(self) -> None:
		if self.monitor:
			self.monitor.stop()
			self.monitor.wait(1000)
			self.monitor = None

	def _on_snapshot(self) -> None:
		if self.history:
			self.snapshotTaken.emit(self.history[-1])

	def _on_sampled(self, sample: MemorySample) -> None:
		self.history.append(sample)
		self._update_summary(sample)
		self._update_plot()

	def _update_summary(self, sample: MemorySample) -> None:
		def fmt(b: int) -> str:
			return f"{b / (1024*1024):.1f} MB"

		self.summary_label.setText(
			f"RSS: {fmt(sample.rss)} | PSS: {fmt(sample.pss)} | USS: {fmt(sample.uss)} | "
			f"Swap: {fmt(sample.swap)} | VMS: {fmt(sample.vms)} | 线程: {sample.threads} | CPU: {sample.cpu_percent:.1f}%"
		)

	def _update_plot(self) -> None:
		if not self.history:
			self.rss_curve.setData([], [])
			self.pss_curve.setData([], [])
			return
		t0 = self.history[0].timestamp
		xs = [s.timestamp - t0 for s in self.history]
		rss_mb = [s.rss / (1024 * 1024) for s in self.history]
		pss_mb = [s.pss / (1024 * 1024) for s in self.history]
		self.rss_curve.setData(xs, rss_mb)
		self.pss_curve.setData(xs, pss_mb)


class MapsTab(QWidget):
	def __init__(self) -> None:
		super().__init__()
		self.pid: Optional[int] = None
		self._build_ui()

	def _build_ui(self) -> None:
		layout = QVBoxLayout(self)

		controls = QHBoxLayout()
		self.refresh_btn = QPushButton("读取内存映射 (smaps)")
		self.export_btn = QPushButton("导出 CSV")
		controls.addWidget(self.refresh_btn)
		controls.addWidget(self.export_btn)
		controls.addStretch(1)
		layout.addLayout(controls)

		self.summary_label = QLabel("")
		layout.addWidget(self.summary_label)

		self.table = QTableWidget()
		self.table.setColumnCount(8)
		self.table.setHorizontalHeaderLabels([
			"类型",
			"Size(MB)",
			"RSS(MB)",
			"PSS(MB)",
			"私有(MB)",
			"共享(MB)",
			"Swap(MB)",
			"路径/名称",
		])
		self.table.horizontalHeader().setStretchLastSection(True)
		layout.addWidget(self.table, stretch=1)

		self.refresh_btn.clicked.connect(self._on_refresh)
		self.export_btn.clicked.connect(self._on_export)

	def attach_to_pid(self, pid: int) -> None:
		self.pid = pid
		self.table.setRowCount(0)
		self.summary_label.setText("")

	def _on_refresh(self) -> None:
		if not self.pid:
			QMessageBox.warning(self, "提示", "请先选择进程")
			return
		try:
			regions = read_smaps(self.pid)
			totals, grouped = aggregate_smaps(self.pid, regions)
		except PermissionError:
			QMessageBox.critical(self, "权限不足", "无法读取 /proc/<pid>/smaps。请尝试使用 root 或选择同一用户的进程。")
			return
		except FileNotFoundError:
			QMessageBox.information(self, "提示", "进程已退出")
			return
		except Exception as exc:
			QMessageBox.critical(self, "错误", f"读取失败: {exc}")
			return

		def kb_to_mb(kb: float) -> float:
			return kb / 1024.0

		self.table.setRowCount(0)
		for group_name, stats in grouped.items():
			row = self.table.rowCount()
			self.table.insertRow(row)
			values = [
				group_name,
				f"{kb_to_mb(stats.get('size_kb', 0)):.1f}",
				f"{kb_to_mb(stats.get('rss_kb', 0)):.1f}",
				f"{kb_to_mb(stats.get('pss_kb', 0)):.1f}",
				f"{kb_to_mb(stats.get('private_kb', 0)):.1f}",
				f"{kb_to_mb(stats.get('shared_kb', 0)):.1f}",
				f"{kb_to_mb(stats.get('swap_kb', 0)):.1f}",
				stats.get('sample_path', ''),
			]
			for col, val in enumerate(values):
				item = QTableWidgetItem(val)
				if col != 7:
					item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
				self.table.setItem(row, col, item)

		total_mb = kb_to_mb(totals.get("pss_kb", 0))
		heap_mb = kb_to_mb(grouped.get("[heap]", {}).get("pss_kb", 0)) if "[heap]" in grouped else 0.0
		anon_mb = kb_to_mb(grouped.get("[anon]", {}).get("pss_kb", 0)) if "[anon]" in grouped else 0.0
		file_mb = kb_to_mb(grouped.get("file", {}).get("pss_kb", 0)) if "file" in grouped else 0.0
		self.summary_label.setText(
			f"PSS 合计: {total_mb:.1f} MB | [heap]: {heap_mb:.1f} MB | 匿名: {anon_mb:.1f} MB | 文件映射: {file_mb:.1f} MB"
		)

	def _on_export(self) -> None:
		if self.table.rowCount() == 0:
			QMessageBox.information(self, "提示", "没有数据可导出，请先读取 smaps")
			return
		path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", os.path.expanduser("~/smaps.csv"), "CSV Files (*.csv)")
		if not path:
			return
		try:
			with open(path, "w", encoding="utf-8") as f:
				headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
				f.write(",".join(headers) + "\n")
				for row in range(self.table.rowCount()):
					values = []
					for col in range(self.table.columnCount()):
						item = self.table.item(row, col)
						values.append((item.text() if item else "").replace(",", " "))
					f.write(",".join(values) + "\n")
			QMessageBox.information(self, "成功", f"已导出到 {path}")
		except Exception as exc:
			QMessageBox.critical(self, "错误", f"导出失败: {exc}")


class DiagnosticsTab(QWidget):
	def __init__(self) -> None:
		super().__init__()
		self.pid: Optional[int] = None
		self._history: Deque[MemorySample] = deque(maxlen=300)
		self._build_ui()

	def _build_ui(self) -> None:
		layout = QVBoxLayout(self)

		self.info = QPlainTextEdit()
		self.info.setReadOnly(True)
		layout.addWidget(self.info)

		controls = QHBoxLayout()
		self.analyze_btn = QPushButton("基于历史样本分析")
		controls.addWidget(self.analyze_btn)
		controls.addStretch(1)
		layout.addLayout(controls)

		self.analyze_btn.clicked.connect(self._on_analyze)

	def attach_to_pid(self, pid: int) -> None:
		self.pid = pid
		self._history.clear()
		self.info.setPlainText("")

	def push_sample(self, sample: MemorySample) -> None:
		self._history.append(sample)
		# Auto-refresh lightweight diagnostics
		self._render_diagnostics(auto=True)

	def _on_analyze(self) -> None:
		self._render_diagnostics(auto=False)

	def _render_diagnostics(self, auto: bool) -> None:
		if not self._history:
			return
		text_lines: List[str] = []
		hist = list(self._history)
		rss_mb = [s.rss / (1024 * 1024) for s in hist]
		pss_mb = [s.pss / (1024 * 1024) for s in hist]
		xs = [s.timestamp - hist[0].timestamp for s in hist]

		def slope(xs: List[float], ys: List[float]) -> float:
			if len(xs) < 2:
				return 0.0
			x0, x1 = xs[0], xs[-1]
			y0, y1 = ys[0], ys[-1]
			dt = max(1e-6, x1 - x0)
			return (y1 - y0) / dt  # MB per second

		rss_slope = slope(xs, rss_mb)
		pss_slope = slope(xs, pss_mb)
		text_lines.append(f"RSS 变化速率: {rss_slope*60:.2f} MB/分钟")
		text_lines.append(f"PSS 变化速率: {pss_slope*60:.2f} MB/分钟")

		if len(hist) >= 15:
			window = 10
			recent = hist[-window:]
			recent_rss = [s.rss / (1024 * 1024) for s in recent]
			increasing = all(b >= a for a, b in zip(recent_rss, recent_rss[1:]))
			if increasing and rss_slope * 60 > 5.0:
				text_lines.append("疑似内存泄漏/未释放：最近一段时间 RSS 持续上升且速率较高")

		try:
			totals, grouped = aggregate_smaps(self.pid)
			pss_total = totals.get("pss_kb", 0.0)
			heap = grouped.get("[heap]", {}).get("pss_kb", 0.0)
			anon = grouped.get("[anon]", {}).get("pss_kb", 0.0)
			file_b = grouped.get("file", {}).get("pss_kb", 0.0)
			if pss_total > 0:
				heap_ratio = heap / pss_total
				anon_ratio = anon / pss_total
				file_ratio = file_b / pss_total
				text_lines.append(
					f"PSS 构成: heap {heap_ratio*100:.1f}%, 匿名 {anon_ratio*100:.1f}%, 文件映射 {file_ratio*100:.1f}%"
				)
				if heap_ratio + anon_ratio > 0.7:
					text_lines.append("大部分内存为私有匿名/堆内存，可能由堆对象增长引起。")
				if file_ratio > 0.5:
					text_lines.append("内存主要由文件映射占用，关注 mmap/内存映射文件的使用。")
		except Exception:
			pass

		if shutil.which("memray") and self.pid is not None:
			# Only hint if likely Python
			try:
				cmd = " ".join(psutil.Process(self.pid).cmdline()).lower()
				if "python" in cmd:
					text_lines.append("检测到 Python 进程且已安装 Memray，可在“Python 分析”页进行深度分配栈分析。")
			except Exception:
				pass

		self.info.setPlainText("\n".join(text_lines))


class PythonTab(QWidget):
	def __init__(self) -> None:
		super().__init__()
		self.pid: Optional[int] = None
		self._build_ui()
		self._memray_proc: Optional[subprocess.Popen] = None
		self._memray_output: Optional[str] = None
		self._capture_path: Optional[str] = None

	def _build_ui(self) -> None:
		layout = QVBoxLayout(self)

		self.status_label = QLabel("此功能针对 Python 进程（需要已安装 memray）")
		layout.addWidget(self.status_label)

		controls = QHBoxLayout()
		self.capture_duration = QSpinBox()
		self.capture_duration.setRange(5, 600)
		self.capture_duration.setValue(20)
		self.capture_duration.setSuffix(" 秒")
		controls.addWidget(QLabel("采样时长:"))
		controls.addWidget(self.capture_duration)

		self.attach_btn = QPushButton("附加并采样")
		self.open_summary_btn = QPushButton("打开现有 summary")
		controls.addWidget(self.attach_btn)
		controls.addWidget(self.open_summary_btn)
		controls.addStretch(1)
		layout.addLayout(controls)

		self.output_view = QPlainTextEdit()
		self.output_view.setReadOnly(True)
		layout.addWidget(self.output_view, stretch=1)

		self.attach_btn.clicked.connect(self._on_attach)
		self.open_summary_btn.clicked.connect(self._on_open_summary)

	def attach_to_pid(self, pid: int) -> None:
		self.pid = pid
		self.status_label.setText(f"目标 PID: {pid} | memray: {'可用' if shutil.which('memray') else '未安装'}")

	def _on_open_summary(self) -> None:
		path, _ = QFileDialog.getOpenFileName(self, "选择 memray 数据或文本", os.path.expanduser("~"), "所有文件 (*.*)")
		if not path:
			return
		text = ""
		try:
			if path.endswith(".bin") and shutil.which("memray"):
				result = subprocess.run(["memray", "summary", "-f", "text", path], capture_output=True, text=True)
				text = result.stdout or result.stderr
			else:
				with open(path, "r", encoding="utf-8", errors="ignore") as f:
					text = f.read()
		except Exception as exc:
			text = f"读取失败: {exc}"
		self.output_view.setPlainText(text)

	def _on_attach(self) -> None:
		if not self.pid:
			QMessageBox.warning(self, "提示", "请先选择进程")
			return
		if not shutil.which("memray"):
			QMessageBox.information(
				self,
				"未检测到 memray",
				"未安装 memray。请运行: pip install memray\n该功能仅适用于 CPython 进程。",
			)
			return
		# Try to detect python process
		try:
			cmdline = " ".join(psutil.Process(self.pid).cmdline()).lower()
			if "python" not in cmdline:
				ret = QMessageBox.question(self, "确认", "目标可能不是 Python 进程，仍要尝试附加？")
				if ret != QMessageBox.Yes:
					return
		except Exception:
			pass

		duration = int(self.capture_duration.value())
		out_path = os.path.join("/tmp", f"memray_{self.pid}_{int(time.time())}.bin")
		self._capture_path = out_path
		self.output_view.setPlainText(f"开始 memray attach，时长 {duration}s，输出: {out_path}\n")

		def worker() -> None:
			try:
				# Start memray attach; it runs until interrupted. We'll terminate after duration.
				proc = subprocess.Popen(["memray", "attach", "-o", out_path, str(self.pid)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
				self._memray_proc = proc
				start_t = time.time()
				while True:
					if proc.stdout is not None:
						line = proc.stdout.readline()
						if line:
							self._append_output(line)
					if time.time() - start_t >= duration:
						break
					if proc.poll() is not None:
						break
				# Try to terminate profiling
				if proc.poll() is None:
					try:
						proc.terminate()
						try:
							proc.wait(timeout=5)
						except subprocess.TimeoutExpired:
							proc.kill()
					except Exception:
						pass
				# Generate summary
				if shutil.which("memray") and os.path.exists(out_path):
					res = subprocess.run(["memray", "summary", "-f", "text", out_path], capture_output=True, text=True)
					self._append_output("\n========== Memray Summary =========\n")
					self._append_output(res.stdout or res.stderr)
			except Exception as exc:
				self._append_output(f"memray 失败: {exc}\n")

		threading.Thread(target=worker, daemon=True).start()

	def _append_output(self, text: str) -> None:
		self.output_view.appendPlainText(text)


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("进程内存分析助手")
		self.resize(1200, 800)
		try:
			self.setWindowIcon(QIcon.fromTheme("utilities-system-monitor"))
		except Exception:
			pass

		self.current_pid: Optional[int] = None

		central = QWidget()
		self.setCentralWidget(central)
		main_layout = QHBoxLayout(central)

		splitter = QSplitter(Qt.Horizontal)
		main_layout.addWidget(splitter)

		# Left panel: process list
		left = QWidget()
		left_layout = QVBoxLayout(left)

		search_row = QHBoxLayout()
		self.search_edit = QLineEdit()
		self.search_edit.setPlaceholderText("搜索进程名/PID/用户…")
		self.only_python_cb = QCheckBox("仅 Python")
		self.refresh_btn = QPushButton("刷新")
		search_row.addWidget(self.search_edit)
		search_row.addWidget(self.only_python_cb)
		search_row.addWidget(self.refresh_btn)
		left_layout.addLayout(search_row)

		self.proc_table = QTableWidget()
		self.proc_table.setColumnCount(5)
		self.proc_table.setHorizontalHeaderLabels(["PID", "名称", "用户", "RSS(MB)", "CPU%"])
		self.proc_table.horizontalHeader().setStretchLastSection(True)
		left_layout.addWidget(self.proc_table)

		splitter.addWidget(left)

		# Right panel: tabs
		right = QWidget()
		right_layout = QVBoxLayout(right)

		self.tabs = QTabWidget()
		self.overview_tab = OverviewTab()
		self.maps_tab = MapsTab()
		self.diag_tab = DiagnosticsTab()
		self.python_tab = PythonTab()

		self.tabs.addTab(self.overview_tab, "概览")
		self.tabs.addTab(self.maps_tab, "映射")
		self.tabs.addTab(self.diag_tab, "诊断")
		self.tabs.addTab(self.python_tab, "Python 分析")

		right_layout.addWidget(self.tabs)
		splitter.addWidget(right)
		splitter.setSizes([400, 800])

		# connections
		self.refresh_btn.clicked.connect(self.refresh_processes)
		self.search_edit.textChanged.connect(lambda _: self.refresh_processes())
		self.only_python_cb.stateChanged.connect(lambda _: self.refresh_processes())
		self.proc_table.itemSelectionChanged.connect(self._on_process_selected)
		self.overview_tab.snapshotTaken.connect(lambda s: self._on_snapshot_taken(s))
		# forward samples to diagnostics
		def forward_sample(sample: MemorySample) -> None:
			self.diag_tab.push_sample(sample)
		self.overview_tab._on_sampled = lambda s: (OverviewTab._on_sampled(self.overview_tab, s), forward_sample(s))

		# Initial load
		self.refresh_processes()

	def refresh_processes(self) -> None:
		query = self.search_edit.text().strip().lower()
		only_py = self.only_python_cb.isChecked()

		processes: List[Tuple[int, str, str, float, float]] = []
		for proc in psutil.process_iter(["pid", "name", "username", "memory_info", "cmdline"]):
			try:
				pid = proc.info.get("pid")
				name = proc.info.get("name") or ""
				user = proc.info.get("username") or ""
				rss = getattr(proc.info.get("memory_info"), "rss", 0) or 0
				cmdline_list = proc.info.get("cmdline") or []
				cmdline = " ".join(cmdline_list).lower()
				if only_py and ("python" not in (name.lower()) and "python" not in cmdline):
					continue
				if query:
					if not (
						query in name.lower()
						or query in user.lower()
						or query in str(pid)
						or (query in cmdline)
					):
						continue
				processes.append((pid, name, user, rss / (1024 * 1024), 0.0))
			except (psutil.NoSuchProcess, psutil.AccessDenied):
				continue
			except Exception:
				continue

		processes.sort(key=lambda x: x[3], reverse=True)

		self.proc_table.setRowCount(0)
		for pid, name, user, rss_mb, cpu in processes[:300]:
			row = self.proc_table.rowCount()
			self.proc_table.insertRow(row)
			for col, val in enumerate([pid, name, user, f"{rss_mb:.1f}", f"{cpu:.1f}"]):
				item = QTableWidgetItem(str(val))
				if col in (0, 3, 4):
					item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
				self.proc_table.setItem(row, col, item)

	def _on_process_selected(self) -> None:
		items = self.proc_table.selectedItems()
		if not items:
			return
		row = items[0].row()
		pid_item = self.proc_table.item(row, 0)
		if not pid_item:
			return
		try:
			pid = int(pid_item.text())
		except ValueError:
			return
		self.current_pid = pid
		self.overview_tab.attach_to_pid(pid)
		self.maps_tab.attach_to_pid(pid)
		self.diag_tab.attach_to_pid(pid)
		self.python_tab.attach_to_pid(pid)

	def _on_snapshot_taken(self, sample: MemorySample) -> None:
		def fmt(b: int) -> str:
			return f"{b/(1024*1024):.1f} MB"
		QMessageBox.information(
			self,
			"快照",
			f"时间: {time.strftime('%H:%M:%S', time.localtime(sample.timestamp))}\n"
			f"RSS: {fmt(sample.rss)}\nPSS: {fmt(sample.pss)}\nUSS: {fmt(sample.uss)}\nSwap: {fmt(sample.swap)}\nVMS: {fmt(sample.vms)}\n线程: {sample.threads}\nCPU: {sample.cpu_percent:.1f}%",
		)


def main() -> None:
	app = QApplication(sys.argv)
	pg.setConfigOptions(antialias=True)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()