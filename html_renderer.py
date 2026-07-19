"""HTML report generation.

Behaviour is preserved exactly from the original synchronous version
(same MathJax protection regexes, same markdown extensions, same
template/CSS) but the output is produced as an in-memory string /
BytesIO buffer instead of being written to disk.
"""

from __future__ import annotations

import io
import re

import markdown

_MATH_PLACEHOLDER = "MATHJAXBLOCK{idx}END"

# الگوهای رایج LaTeX — ChatGPT اغلب \( \) و \[ \] می‌فرستد که markdown خراب می‌کند
_MATH_PATTERNS = [
    (r"\$\$([\s\S]+?)\$\$", "display"),
    (r"\\\[([\s\S]+?)\\\]", "display"),
    (
        r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|cases)\}"
        r"([\s\S]+?)\\end\{\1\}",
        "env",
    ),
    (r"```(?:latex|math)?\s*\n([\s\S]+?)```", "display"),
    (r"\\\(([\s\S]+?)\\\)", "inline"),
    (r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", "inline"),
]


def _stash_math(match: re.Match, kind: str, placeholders: list[str]) -> str:
    idx = len(placeholders)
    if kind == "env":
        env_name = match.group(1)
        body = match.group(2).strip()
        tex = f"$$\n\\begin{{{env_name}}}\n{body}\n\\end{{{env_name}}}\n$$"
    elif kind == "display":
        tex = f"$$\n{match.group(1).strip()}\n$$"
    else:
        tex = f"${match.group(1).strip()}$"
    placeholders.append(tex)
    return _MATH_PLACEHOLDER.format(idx=idx)


def _protect_math(text: str) -> tuple[str, list[str]]:
    placeholders: list[str] = []
    for pattern, kind in _MATH_PATTERNS:
        text = re.sub(pattern, lambda m, k=kind: _stash_math(m, k, placeholders), text)
    return text, placeholders


def _restore_math(html: str, placeholders: list[str]) -> str:
    for idx, tex in enumerate(placeholders):
        html = html.replace(_MATH_PLACEHOLDER.format(idx=idx), tex)
    return html


def markdown_with_math(text: str) -> str:
    """فرمول‌ها را قبل از markdown محافظت می‌کند تا _ و \\ خراب نشوند."""
    protected, placeholders = _protect_math(text)
    html = markdown.markdown(protected, extensions=["nl2br"])
    return _restore_math(html, placeholders)


_HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>پاسخ‌نامه هوش مصنوعی</title>

    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true,
        packages: {{'[+]': ['ams']}}
      }},
      chtml: {{
        scale: 1,
        minScale: 0.55,
        matchFontHeight: false
      }},
      options: {{
        renderActions: {{
          addMenu: [0, '', '']
        }}
      }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        * {{ box-sizing: border-box; }}
        html, body {{ max-width: 100%; overflow-x: hidden; }}
        body {{
            font-family: 'Tahoma', 'Arial', sans-serif;
            line-height: 1.8;
            padding: 16px;
            background-color: #f1f5f9;
            color: #1e293b;
            direction: rtl;
            font-size: 15px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h2 {{
            text-align: center;
            color: #1e3a8a;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 12px;
            margin-bottom: 24px;
            font-size: 1.2rem;
        }}
        .columns {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: stretch;
        }}
        .box {{
            flex: 1 1 320px;
            min-width: 0;
            max-width: 100%;
            border: 1px solid #e2e8f0;
            padding: 16px;
            border-radius: 12px;
            background: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .gemini {{ border-right: 6px solid #1a73e8; }}
        .claude {{ border-right: 6px solid #d97706; }}
        .gpt {{ border-right: 6px solid #10a37f; }}
        h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.05rem;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
        }}
        .gemini h3 {{ color: #1a73e8; }}
        .claude h3 {{ color: #d97706; }}
        .gpt h3 {{ color: #10a37f; }}

        .content-area {{
            white-space: pre-wrap;
            overflow-wrap: normal;
            word-break: normal;
            font-size: 0.95rem;
            max-width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 6px;
        }}
        /* هر خط طولانی (فرمول یا محاسبه بدون فاصله) به‌جای شکستن حروف یا
           بیرون‌زدگی از قاب، خودش به‌صورت مستقل اسکرول افقی می‌گیرد */
        .content-area p,
        .content-area li,
        .content-area blockquote {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            max-width: 100%;
            padding-bottom: 4px;
        }}
        .content-area::-webkit-scrollbar,
        .content-area p::-webkit-scrollbar,
        .content-area li::-webkit-scrollbar,
        mjx-container::-webkit-scrollbar {{
            height: 6px;
        }}
        .content-area::-webkit-scrollbar-thumb,
        .content-area p::-webkit-scrollbar-thumb,
        .content-area li::-webkit-scrollbar-thumb,
        mjx-container::-webkit-scrollbar-thumb {{
            background: #cbd5e1;
            border-radius: 4px;
        }}
        .content-area ul, .content-area ol {{
            padding-right: 20px;
            padding-left: 0;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
        .content-area li {{
            margin-bottom: 8px;
        }}
        .content-area img {{
            max-width: 100%;
            height: auto;
        }}
        .content-area table {{
            display: block;
            max-width: 100%;
            overflow-x: auto;
            border-collapse: collapse;
            white-space: normal;
        }}
        .content-area code, .content-area pre {{
            overflow-x: auto;
            max-width: 100%;
            white-space: pre;
            word-break: normal;
        }}

        mjx-container {{
            max-width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
        }}
        mjx-container[display="true"] {{
            display: block !important;
            margin: 0.6em 0 !important;
            padding-bottom: 4px;
        }}

        @media (max-width: 900px) {{
            .columns {{
                flex-direction: column;
            }}
            .box {{
                flex: 1 1 auto;
            }}
        }}

        @media (max-width: 480px) {{
            body {{ padding: 10px; font-size: 14px; }}
            h2 {{ font-size: 1.05rem; margin-bottom: 16px; }}
            .box {{ padding: 12px; border-radius: 10px; }}
            h3 {{ font-size: 0.95rem; margin-bottom: 10px; }}
            .content-area {{ font-size: 0.88rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>📄 پاسخ‌نامه تشریحی هوش مصنوعی</h2>
        <div class="columns">
            <div class="box gemini">
                <h3>🤖 پاسخ مدل Gemini (Google)</h3>
                <div class="content-area">{gemini_html}</div>
            </div>
            <div class="box claude">
                <h3>✨ پاسخ مدل Claude (Anthropic)</h3>
                <div class="content-area">{claude_html}</div>
            </div>
            <div class="box gpt">
                <h3>🧠 پاسخ مدل ChatGPT (OpenAI)</h3>
                <div class="content-area">{gpt_html}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""


def render_report_html(gemini_answer: str, claude_answer: str, gpt_answer: str) -> str:
    """Build the full HTML string for the three-column report."""
    gemini_html = markdown_with_math(gemini_answer)
    claude_html = markdown_with_math(claude_answer)
    gpt_html = markdown_with_math(gpt_answer)
    return _HTML_TEMPLATE.format(
        gemini_html=gemini_html, claude_html=claude_html, gpt_html=gpt_html
    )


def render_report_buffer(gemini_answer: str, claude_answer: str, gpt_answer: str) -> io.BytesIO:
    """Same as `render_report_html` but returns a ready-to-upload
    in-memory buffer, so no temp file ever touches disk."""
    html = render_report_html(gemini_answer, claude_answer, gpt_answer)
    buffer = io.BytesIO(html.encode("utf-8"))
    buffer.seek(0)
    return buffer
