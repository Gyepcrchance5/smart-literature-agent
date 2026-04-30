"""[Phase 2 stub] 期刊网站 HTML 页面 → 公式提取。

=============================================================
本模块当前**未实现**。在部署到可访问期刊网站的校园网电脑后
按下列步骤启用：
=============================================================

职责：从 IEEE Xplore / ScienceDirect / Nature / Science 的
在线阅读页提取论文内容和公式。这些页面里公式通常以：
  - MathJax（`<span class="mjx-math">` 或 `<script type="math/tex">`）
  - MathML（`<math>...</math>`）
两种形式出现。

建议工具链：
  pip install beautifulsoup4 lxml selectolax
  pip install mathml-to-latex  # MathML → LaTeX 转换

实装步骤（每个期刊一个 adapter）：
  1. 识别期刊：按 URL 的 host 分派（ieee.org / sciencedirect.com / nature.com / science.org）
  2. 登录态：复用浏览器 cookie 或 requests session（校园网 IP 白名单 + 会话）
  3. 抓取 HTML → 用 adapter 提取 <main> 正文 + 公式节点
  4. 把 MathML 或 MathJax → LaTeX 字符串
  5. 调 formula_handler.extract_from_latex 复用解析器（或直接返回 Formula 列表）

常见坑：
  - JS 渲染的页面需要 Playwright / Selenium
  - IEEE 某些旧论文仍是扫描 PDF（这时落回 pdf_handler 路线）
"""
from __future__ import annotations

from utils import get_logger

log = get_logger("html_handler")


def extract_from_html(url: str, **kwargs) -> list:
    raise NotImplementedError(
        "HTML 公式提取未实装（Phase 2）。\n"
        "部署到校园网电脑后按 src/html_handler.py 顶部的 docstring 指引实现期刊 adapter。"
    )
