"""[Phase 2 stub] PDF → 公式提取。

=============================================================
本模块当前**未实现**。在部署到可访问 IEEE / ScienceDirect 的
校园网电脑后按下列步骤启用：
=============================================================

推荐使用 MinerU（OpenDataLab 维护，国内友好）：

  pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
  magic-pdf --help

如想换 Marker（速度更快、资源占用低）：

  pip install marker-pdf

实装指引：
  1. 用选定工具把 PDF 转成 markdown（公式保留 LaTeX）
  2. 调用 formula_handler.extract_from_latex(markdown_text) 复用解析器
  3. 返回 list[Formula]，保持与 arxiv 路线一致的 interface

GPU 推荐：CUDA 11+，不然 CPU 上每页 30-60 秒。
"""
from __future__ import annotations

from pathlib import Path

from utils import get_logger

log = get_logger("pdf_handler")


def extract_from_pdf(pdf_path: str | Path, **kwargs) -> list:
    raise NotImplementedError(
        "PDF 公式提取未实装（Phase 2）。\n"
        "部署到校园网电脑后按 src/pdf_handler.py 顶部的 docstring 指引安装 MinerU 或 Marker。"
    )
