# Analysis_Memory

## 进程内存分析助手（PyQt）

- 实时监控进程 RSS/PSS/USS/Swap/VMS、CPU%、线程数
- 解析 `/proc/<pid>/smaps`，分组展示 heap/匿名/文件映射等并导出 CSV
- 提供基于趋势与构成的诊断提示
- 可选：若安装 `memray`，可对 Python 进程进行分配采样并生成 summary

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行
```bash
python -m app.main
```

提示：读取他人进程的 smaps 可能需要相同用户或 root 权限；PSS/USS 依赖内核与权限，无法获取时会自动回退。