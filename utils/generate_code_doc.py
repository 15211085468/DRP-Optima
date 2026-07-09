#!/usr/bin/env python3
"""
DRP-Optima 代码文档生成器 (聚焦核心业务逻辑版 v3)
自动扫描项目文件，生成深色主题的 HTML 代码文档。
聚焦于核心业务逻辑和可维护性，排除缓存、虚拟环境、构建产物、IDE 配置等无关文件。
"""

import os
import sys
import ast
import html as html_lib
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── 项目根目录 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 文件类型配置 ────────────────────────────────────────────
# 支持的文件扩展名及其语言标识（用于语法高亮）
# 只包含核心业务相关的文件类型
SUPPORTED_EXTENSIONS = {
    # Python 源码（核心业务逻辑）
    ".py": "python",
    # 配置文件（业务配置）
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    # 文档（项目文档）
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    # Web（项目 Web 界面）
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".js": "javascript",
    # 脚本（部署/自动化脚本）
    ".sh": "bash",
    ".bash": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
}

# 无扩展名的特殊文件（必须包含，属于业务逻辑）
INCLUDE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "docker-compose.yml",
    "docker-compose.yaml",
}

# 排除的目录名（缓存、虚拟环境、构建产物、IDE 配置、训练输出）
EXCLUDE_DIRS = {
    # 缓存目录
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    # 版本控制
    ".git",
    # WorkBuddy 内部目录
    ".workbuddy",
    # 虚拟环境
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    ".virtualenvs",
    "site-packages",
    # 构建产物
    "build",
    "dist",
    "wheels",
    # IDE 配置
    ".idea",
    ".vscode",
    ".eclipse",
    # 数据目录（不包含数据文件）
    "data/processed",
    "data/raw",
    "data/interim",
    "data/external",
    # 训练输出（不属于核心业务逻辑）
    "lightning_logs",
    "tensorboard_logs",
    "checkpoints",
    "output",
    "test_output",
    # 归档目录（旧版本文件）
    "archive",
}

# 排除的文件扩展名（即使主扩展名在 SUPPORTED_EXTENSIONS 中）
EXCLUDE_EXTENSIONS = {
    # 字节码文件
    ".pyc",
    ".pyo",
    ".pyd",
    # 锁文件（依赖锁定，不审查具体哈希值）
    ".lock",
    # 日志文件
    ".log",
    # 数据库文件
    ".db",
    ".sqlite3",
    # 二进制资源
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".mp4",
    ".avi",
    ".pkl",
    ".pickle",
    # 敏感文件
    ".key",
    ".pem",
    ".crt",
    ".cert",
}

# 排除的文件名（敏感信息/本地配置）
EXCLUDE_FILENAMES = {
    ".env",
    ".secrets",
    "config.local.yaml",
    "config.local.yml",
    "config.local.json",
    "poetry.lock",
    "Pipfile.lock",
}

# 大文件阈值（超过此大小的文件不显示完整源码）
LARGE_FILE_THRESHOLD = 50 * 1024  # 50KB


# ── 文件扫描 ────────────────────────────────────────────────

def scan_all_files(project_root: Path) -> list:
    """
    递归扫描项目根目录下的所有支持的文件。
    返回列表：[(relative_path, absolute_path, file_type), ...]
    """
    results = []
    
    for file_path in project_root.rglob("*"):
        # 跳过目录
        if not file_path.is_file():
            continue
        
        # 获取文件名和扩展名
        filename = file_path.name
        ext = file_path.suffix.lower()
        ext_with_dot = file_path.suffix  # 包含点号，如 ".py"
        
        # 检查是否在排除的目录中
        file_str = str(file_path)
        if any(exc_dir in file_str for exc_dir in EXCLUDE_DIRS):
            continue
        
        # 检查是否为排除的文件名
        if filename in EXCLUDE_FILENAMES:
            continue
        
        # 检查是否为排除的扩展名
        if ext_with_dot.lower() in EXCLUDE_EXTENSIONS:
            continue
        
        # 检查文件扩展名或文件名是否在支持列表中
        rel_path = file_path.relative_to(project_root)
        rel_path_str = str(rel_path).replace("\\", "/")
        
        # 情况 1：有扩展名且在 SUPPORTED_EXTENSIONS 中
        if ext_with_dot.lower() in SUPPORTED_EXTENSIONS:
            file_type = SUPPORTED_EXTENSIONS[ext_with_dot.lower()]
            results.append((rel_path_str, file_path, file_type))
            continue
        
        # 情况 2：无扩展名，但在 INCLUDE_FILENAMES 中
        if filename in INCLUDE_FILENAMES:
            # 根据文件名判断语言类型
            if filename == "Dockerfile":
                file_type = "dockerfile"
            elif filename == "Makefile":
                file_type = "makefile"
            elif "docker-compose" in filename:
                file_type = "yaml"
            else:
                file_type = "text"
            results.append((rel_path_str, file_path, file_type))
            continue
        
        # 情况 3：都不匹配，跳过
        continue
    
    # 按路径排序
    results.sort(key=lambda x: x[0].lower())
    
    return results


def read_file_safe(file_path: Path) -> str:
    """安全读取文件内容（尝试多种编码）。"""
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    
    # 如果所有编码都失败，返回空字符串
    return ""


# ── Python 文件解析 ─────────────────────────────────────────

def parse_python_file(filepath: Path) -> dict:
    """解析单个 .py 文件，提取模块文档、类、函数信息。"""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        # 语法错误，但仍返回基本信息
        return {
            "path": str(filepath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "filename": filepath.name,
            "line_count": len(source.splitlines()),
            "module_docstring": "",
            "classes": [],
            "functions": [],
            "source": source,
            "parse_error": "SyntaxError",
        }

    rel_path = filepath.relative_to(PROJECT_ROOT)
    lines = source.splitlines()

    result = {
        "path": str(rel_path).replace("\\", "/"),
        "filename": filepath.name,
        "line_count": len(lines),
        "module_docstring": ast.get_docstring(tree) or "",
        "classes": [],
        "functions": [],
        "source": source,
        "parse_error": None,
    }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            cls_info = _parse_class(node, lines)
            result["classes"].append(cls_info)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = _parse_function(node, lines)
            result["functions"].append(func_info)

    return result


def _parse_class(node: ast.ClassDef, lines: list) -> dict:
    """提取类的信息。"""
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(ast.unparse(base))
        else:
            bases.append(ast.unparse(base))

    methods = []
    class_vars = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_parse_function(child, lines))
        elif isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            if isinstance(target, ast.Name):
                class_vars.append({
                    "name": target.id,
                    "line": child.lineno,
                    "value": ast.unparse(child.value) if child.value else "",
                })

    return {
        "name": node.name,
        "bases": bases,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "docstring": ast.get_docstring(node) or "",
        "methods": methods,
        "class_vars": class_vars,
    }


def _parse_function(node, lines: list) -> dict:
    """提取函数/方法的信息。"""
    args_list = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        args_list.append(arg_str)

    # 默认值
    defaults = node.args.defaults
    n_args = len(node.args.args)
    n_defaults = len(defaults)
    for i, default in enumerate(defaults):
        idx = n_args - n_defaults + i
        try:
            args_list[idx] += f" = {ast.unparse(default)}"
        except Exception:
            pass

    # *args
    if node.args.vararg:
        args_list.append(f"*{node.args.vararg.arg}")

    # **kwargs
    if node.args.kwarg:
        args_list.append(f"**{node.args.kwarg.arg}")

    # 返回类型
    returns = ""
    if node.returns:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass

    sig = f"({', '.join(args_list)}){returns}"

    # decorators
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append(f"@{ast.unparse(dec)}")
        except Exception:
            pass

    return {
        "name": node.name,
        "decorators": decorators,
        "signature": sig,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "docstring": ast.get_docstring(node) or "",
    }


# ── HTML 生成（增量写入）─────────────────────────────────────

def esc(text: str) -> str:
    """HTML 转义。"""
    return html_lib.escape(str(text))


def generate_html_incremental(all_files_data: list, stats: dict, output_path: Path):
    """
    增量生成 HTML 文档（边生成边写入，避免内存爆炸）。
    """
    
    # 按目录分组
    grouped = defaultdict(list)
    for file_data in all_files_data:
        parts = file_data["path"].split("/")
        if len(parts) == 1:
            group = "根目录"
        else:
            group = parts[0]  # 第一级目录名
        grouped[group].append(file_data)
    
    # 按字母顺序排序组
    sorted_groups = sorted(grouped.keys(), key=lambda x: (x != "根目录", x.lower()))
    
    # 生成目录（TOC）
    toc_items = []
    for group_name in sorted_groups:
        group_files = grouped[group_name]
        group_id = group_name.replace("/", "_").replace(".", "_").replace(" ", "_")
        toc_items.append(
            f'<li><a href="#{esc(group_id)}" class="toc-link">{esc(group_name)} ({len(group_files)} 文件)</a></li>'
        )
    
    toc_html = "\n".join(toc_items)
    
    # 开始写入 HTML 文件
    with open(output_path, "w", encoding="utf-8") as f:
        # 写入 HTML 头部
        f.write(HTML_HEADER.format(
            toc=toc_html,
            total_files=stats["total_files"],
            total_classes=stats["total_classes"],
            total_functions=stats["total_functions"],
            total_lines=stats["total_lines"],
            gen_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))
        
        # 写入每个组的内容
        total_files_count = sum(len(files) for files in grouped.values())
        processed_count = 0
        
        for group_name in sorted_groups:
            group_files = grouped[group_name]
            group_id = group_name.replace("/", "_").replace(".", "_").replace(" ", "_")
            
            # 组标题
            f.write(f"""
        <div class="module-group" id="{esc(group_id)}">
            <h2 class="group-title">{esc(group_name)} <span class="file-count">({len(group_files)} 文件)</span></h2>
        """)
            
            # 每个文件
            for file_data in group_files:
                # 每 10 个文件打印一次进度
                processed_count += 1
                if processed_count % 10 == 0:
                    progress = processed_count / total_files_count * 100
                    print(f"  生成 HTML 进度: {processed_count}/{total_files_count} ({progress:.1f}%) | 当前: {file_data['path'][:50]}...")
                
                html_section = _render_file_section(file_data)
                f.write(html_section)
            
            f.write("        </div>\n")
        
        # 写入 HTML 尾部
        f.write(HTML_FOOTER.format(
            gen_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
    
    print(f"  生成 HTML 进度: {total_files_count}/{total_files_count} (100.0%)")


def _render_file_section(file_data: dict) -> str:
    """渲染单个文件的文档区段（返回 HTML 字符串）。"""
    path = file_data["path"]
    filename = file_data["filename"]
    line_count = file_data["line_count"]
    file_type = file_data.get("file_type", "text")
    source = file_data.get("source", "")
    file_size = len(source.encode("utf-8"))
    
    # 判断是否为大文件
    is_large = file_size > LARGE_FILE_THRESHOLD
    
    # 如果是 Python 文件且有类/函数信息，渲染详细信息
    if file_data.get("is_python") and file_data.get("parse_error") is None:
        return _render_python_file(file_data, is_large)
    
    # 其他文件：显示完整源码（如果不是大文件）
    doc_html = ""
    if file_data.get("module_docstring"):
        doc_html = f'<div class="module-doc"><pre>{esc(file_data["module_docstring"])}</pre></div>'
    
    source_html = ""
    if source and not is_large:
        # 显示完整源码
        source_html = f'<div class="source-block"><pre><code class="language-{file_type}">{esc(source)}</code></pre></div>'
    elif source and is_large:
        # 大文件：只显示前 100 行
        lines = source.splitlines()
        preview = "\n".join(lines[:100])
        source_html = f'<div class="large-file-warning">⚠️ 文件过大 ({file_size/1024:.1f} KB)，仅显示前 100 行</div>'
        source_html += f'<div class="source-block"><pre><code class="language-{file_type}">{esc(preview)}</code></pre>'
        if len(lines) > 100:
            source_html += f'\n<span class="omitted">... 省略 {len(lines) - 100} 行 ...</span>'
        source_html += '</div>'
    
    # 文件行数统计
    stats_html = f'<span class="file-stats">{line_count} 行 | {file_type}'
    if is_large:
        stats_html += f' | ⚠️ 大文件 ({file_size/1024:.1f} KB)'
    stats_html += '</span>'
    
    return f"""
    <div class="file-section">
        <div class="file-header" onclick="toggleSection(this)">
            <span class="file-icon">📄</span>
            <span class="file-path">{esc(path)}</span>
            {stats_html}
            <span class="toggle-icon">▼</span>
        </div>
        <div class="file-content">
            {doc_html}
            {source_html}
        </div>
    </div>
    """


def _render_python_file(file_data: dict, is_large: bool) -> str:
    """渲染 Python 文件的详细文档。"""
    path = file_data["path"]
    filename = file_data["filename"]
    line_count = file_data["line_count"]
    module_doc = file_data.get("module_docstring", "")
    classes = file_data.get("classes", [])
    functions = file_data.get("functions", [])
    source = file_data.get("source", "")
    parse_error = file_data.get("parse_error")
    
    # 模块 docstring
    doc_html = ""
    if module_doc:
        doc_html = f'<div class="module-doc"><pre>{esc(module_doc)}</pre></div>'
    
    # 解析错误提示
    error_html = ""
    if parse_error:
        error_html = f'<div class="parse-error">⚠️ 解析错误: {esc(parse_error)}</div>'
    
    # 类
    class_html = ""
    if classes:
        class_items = []
        for cls in classes:
            class_items.append(_render_class(cls))
        class_html = f'<div class="classes-section"><h4>类定义 ({len(classes)})</h4>{"".join(class_items)}</div>'
    
    # 模块级函数
    func_html = ""
    if functions:
        func_items = []
        for func in functions:
            func_items.append(_render_function(func))
        func_html = f'<div class="functions-section"><h4>函数 ({len(functions)})</h4>{"".join(func_items)}</div>'
    
    # 完整源码（可折叠，大文件只显示前 100 行）
    source_html = ""
    if source and not is_large:
        source_html = f'<details class="source-details"><summary>完整源码 ({line_count} 行)</summary><div class="source-block"><pre><code class="language-python">{esc(source)}</code></pre></div></details>'
    elif source and is_large:
        lines = source.splitlines()
        preview = "\n".join(lines[:100])
        source_html = f'<details class="source-details"><summary>完整源码 (仅预览前 100 行，共 {line_count} 行)</summary>'
        source_html += f'<div class="large-file-warning">⚠️ 文件过大，仅显示前 100 行</div>'
        source_html += f'<div class="source-block"><pre><code class="language-python">{esc(preview)}</code></pre>'
        if len(lines) > 100:
            source_html += f'\n<span class="omitted">... 省略 {len(lines) - 100} 行 ...</span>'
        source_html += '</div></details>'
    
    # 文件行数统计
    stats_html = f'<span class="file-stats">{line_count} 行'
    if classes:
        stats_html += f' | {len(classes)} 类'
    if functions:
        stats_html += f' | {len(functions)} 函数'
    file_size = len(source.encode("utf-8"))
    if file_size > LARGE_FILE_THRESHOLD:
        stats_html += f' | ⚠️ 大文件 ({file_size/1024:.1f} KB)'
    stats_html += '</span>'
    
    return f"""
    <div class="file-section">
        <div class="file-header" onclick="toggleSection(this)">
            <span class="file-icon">🐍</span>
            <span class="file-path">{esc(path)}</span>
            {stats_html}
            <span class="toggle-icon">▼</span>
        </div>
        <div class="file-content">
            {error_html}
            {doc_html}
            {class_html}
            {func_html}
            {source_html}
        </div>
    </div>
    """


def _render_class(cls: dict) -> str:
    """渲染单个类的文档。"""
    bases_str = ""
    if cls["bases"]:
        bases_str = f'<span class="class-bases">({esc(", ".join(cls["bases"]))})</span>'
    
    doc_html = ""
    if cls["docstring"]:
        doc_html = f'<div class="docstring">{esc(cls["docstring"])}</div>'
    
    # 类变量
    vars_html = ""
    if cls["class_vars"]:
        var_items = []
        for v in cls["class_vars"]:
            var_items.append(
                f'<div class="class-var"><code>{esc(v["name"])}</code> = <code class="value">{esc(v["value"])}</code></div>'
            )
        vars_html = f'<div class="class-vars"><h5>类变量</h5>{"".join(var_items)}</div>'
    
    # 方法
    methods_html = ""
    if cls["methods"]:
        method_items = []
        for m in cls["methods"]:
            method_items.append(_render_function(m, is_method=True))
        methods_html = f'<div class="methods"><h5>方法 ({len(cls["methods"])})</h5>{"".join(method_items)}</div>'
    
    return f"""
    <div class="class-block">
        <div class="class-header" onclick="toggleSection(this)">
            <span class="keyword">class</span> <span class="class-name">{esc(cls["name"])}</span>{bases_str}
            <span class="line-ref">L{cls["line"]}</span>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="class-content">
            {doc_html}
            {vars_html}
            {methods_html}
        </div>
    </div>
    """


def _render_function(func: dict, is_method: bool = False) -> str:
    """渲染单个函数/方法的文档。"""
    decorators_html = ""
    if func["decorators"]:
        dec_items = [f'<span class="decorator">{esc(d)}</span>' for d in func["decorators"]]
        decorators_html = '<div class="decorators">' + " ".join(dec_items) + '</div>'
    
    doc_html = ""
    if func["docstring"]:
        doc_html = f'<div class="docstring">{esc(func["docstring"])}</div>'
    
    kind = "def"
    
    return f"""
    <div class="function-block">
        {decorators_html}
        <div class="function-sig">
            <span class="keyword">{kind}</span> <span class="func-name">{esc(func["name"])}</span><span class="func-params">{esc(func["signature"])}</span>
            <span class="line-ref">L{func["line"]}</span>
        </div>
        {doc_html}
    </div>
    """


# ── HTML 模板（分成头部和尾部）──────────────────────────────

HTML_HEADER = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DRP-Optima 项目代码文档 (核心业务版)</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --bg-hover: #30363d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-blue: #58a6ff;
            --accent-green: #7ee787;
            --accent-orange: #ffa657;
            --accent-purple: #d2a8ff;
            --accent-red: #ff7b72;
            --accent-cyan: #79c0ff;
            --accent-yellow: #f0e68c;
            --link-color: #58a6ff;
            --code-bg: #0d1117;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            font-size: 14px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            gap: 20px;
        }}
        
        /* ── 侧边目录 ── */
        .sidebar {{
            width: 280px;
            flex-shrink: 0;
            position: sticky;
            top: 20px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}
        
        .sidebar h3 {{
            color: var(--accent-blue);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 12px 0 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 8px;
        }}
        
        .sidebar ul {{
            list-style: none;
        }}
        
        .sidebar li {{
            margin-bottom: 2px;
        }}
        
        .toc-link {{
            display: block;
            padding: 6px 12px;
            color: var(--text-secondary);
            text-decoration: none;
            border-radius: 4px;
            font-size: 13px;
            transition: all 0.15s;
        }}
        
        .toc-link:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        
        .toc-link.active {{
            background: var(--bg-tertiary);
            color: var(--accent-blue);
            border-left: 2px solid var(--accent-blue);
        }}
        
        /* ── 主内容区 ── */
        .main-content {{
            flex: 1;
            min-width: 0;
        }}
        
        /* ── 头部 ── */
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #1e1b4b 100%);
            padding: 32px;
            border-radius: 12px;
            margin-bottom: 28px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            border: 1px solid rgba(88,166,255,0.15);
        }}
        
        .header h1 {{
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        
        .header .subtitle {{
            color: #93c5fd;
            font-size: 15px;
            margin-bottom: 16px;
        }}
        
        .header .info {{
            color: #bfdbfe;
            font-size: 13px;
            line-height: 1.8;
        }}
        
        /* ── 统计卡片 ── */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 28px;
        }}
        
        .stat-card {{
            background: var(--bg-secondary);
            padding: 16px 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        
        .stat-card .label {{
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card .value {{
            color: var(--accent-blue);
            font-size: 22px;
            font-weight: 700;
            margin-top: 4px;
        }}
        
        /* ── 模块组 ── */
        .module-group {{
            margin-bottom: 32px;
        }}
        
        .group-title {{
            color: var(--accent-blue);
            font-size: 18px;
            font-weight: 600;
            padding: 10px 0 8px;
            border-bottom: 2px solid var(--accent-blue);
            margin-bottom: 12px;
            letter-spacing: -0.3px;
        }}
        
        .file-count {{
            color: var(--text-muted);
            font-size: 14px;
            font-weight: normal;
        }}
        
        /* ── 文件区段 ── */
        .file-section {{
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}
        
        .file-header {{
            padding: 10px 16px;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.15s;
        }}
        
        .file-header:hover {{
            background: var(--bg-tertiary);
        }}
        
        .file-icon {{
            font-size: 14px;
        }}
        
        .file-path {{
            color: var(--accent-cyan);
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 13px;
            flex: 1;
        }}
        
        .file-stats {{
            color: var(--text-muted);
            font-size: 11px;
        }}
        
        .toggle-icon {{
            color: var(--text-muted);
            font-size: 12px;
            transition: transform 0.2s;
        }}
        
        .file-header.collapsed .toggle-icon {{
            transform: rotate(-90deg);
        }}
        
        .file-content {{
            padding: 0 16px 16px;
            border-top: 1px solid var(--border-color);
        }}
        
        .file-header.collapsed + .file-content {{
            display: none;
        }}
        
        /* ── 模块 docstring ── */
        .module-doc {{
            padding: 12px;
            margin: 8px 0;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border-left: 3px solid var(--accent-green);
        }}
        
        .module-doc pre {{
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 12.5px;
            line-height: 1.5;
            color: var(--text-secondary);
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        /* ── 解析错误 ── */
        .parse-error {{
            padding: 8px 12px;
            margin: 8px 0;
            background: rgba(255, 123, 114, 0.1);
            border-radius: 6px;
            border-left: 3px solid var(--accent-red);
            color: var(--accent-red);
            font-size: 12px;
        }}
        
        /* ── 大文件警告 ── */
        .large-file-warning {{
            padding: 8px 12px;
            margin: 8px 0;
            background: rgba(255, 166, 87, 0.1);
            border-radius: 6px;
            border-left: 3px solid var(--accent-orange);
            color: var(--accent-orange);
            font-size: 12px;
        }}
        
        .omitted {{
            color: var(--text-muted);
            font-style: italic;
        }}
        
        /* ── 类 ── */
        .classes-section, .functions-section {{
            margin-top: 12px;
        }}
        
        .classes-section h4, .functions-section h4, .classes-section h5, .methods h5 {{
            color: var(--text-secondary);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .class-block {{
            margin-bottom: 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}
        
        .class-header {{
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 13px;
        }}
        
        .class-header:hover {{
            background: var(--bg-hover);
        }}
        
        .class-header.collapsed + .class-content {{
            display: none;
        }}
        
        .class-name {{
            color: var(--accent-green);
            font-weight: 600;
        }}
        
        .class-bases {{
            color: var(--accent-orange);
        }}
        
        .class-content {{
            padding: 0 12px 12px;
            border-top: 1px solid var(--border-color);
        }}
        
        .class-vars {{
            margin: 8px 0;
            padding: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
        }}
        
        .class-vars h5 {{
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        
        .class-var {{
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 12px;
            margin-bottom: 4px;
        }}
        
        .class-var code {{
            color: var(--accent-purple);
        }}
        
        .class-var code.value {{
            color: var(--accent-orange);
        }}
        
        /* ── 方法区 ── */
        .methods {{
            margin-top: 8px;
        }}
        
        .methods h5 {{
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        
        /* ── 函数 ── */
        .function-block {{
            padding: 6px 12px;
            margin-bottom: 4px;
            border-radius: 4px;
            background: var(--bg-primary);
        }}
        
        .decorators {{
            margin-bottom: 2px;
        }}
        
        .decorator {{
            color: var(--accent-yellow);
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 12px;
        }}
        
        .function-sig {{
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 12.5px;
        }}
        
        .keyword {{
            color: var(--accent-red);
            font-weight: 600;
        }}
        
        .func-name {{
            color: var(--accent-blue);
            font-weight: 500;
        }}
        
        .func-params {{
            color: var(--text-secondary);
        }}
        
        .line-ref {{
            color: var(--text-muted);
            font-size: 11px;
            margin-left: 8px;
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
        }}
        
        .docstring {{
            padding: 8px 12px;
            margin: 6px 0;
            background: rgba(126, 231, 135, 0.06);
            border-radius: 4px;
            border-left: 2px solid var(--accent-green);
            font-size: 12px;
            color: var(--text-secondary);
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
        }}
        
        /* ── 源码块 ── */
        .source-details {{
            margin-top: 12px;
        }}
        
        .source-details summary {{
            cursor: pointer;
            color: var(--accent-blue);
            font-size: 12px;
            padding: 4px 0;
        }}
        
        .source-block {{
            margin-top: 8px;
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
        }}
        
        .source-block pre {{
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 12px;
            line-height: 1.5;
            color: var(--text-secondary);
        }}
        
        /* ── 滚动条 ── */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: var(--bg-primary);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--bg-hover);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}
        
        /* ── 响应式 ── */
        @media (max-width: 900px) {{
            .container {{
                flex-direction: column;
            }}
            .sidebar {{
                width: 100%;
                position: static;
                max-height: none;
            }}
        }}
        
        /* ── 搜索 ── */
        .search-box {{
            margin-bottom: 16px;
        }}
        
        .search-input {{
            width: 100%;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 13px;
            outline: none;
        }}
        
        .search-input:focus {{
            border-color: var(--accent-blue);
        }}
        
        .search-input::placeholder {{
            color: var(--text-muted);
        }}
        
        /* ── Footer ── */
        .footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 侧边目录 -->
        <nav class="sidebar">
            <h3>目录导航</h3>
            <div class="search-box">
                <input type="text" class="search-input" placeholder="搜索文件/类/函数..." id="searchInput" oninput="filterContent(this.value)">
            </div>
            <ul id="tocList">
                {toc}
            </ul>
        </nav>
        
        <!-- 主内容 -->
        <main class="main-content">
            <div class="header">
                <h1>DRP-Optima 项目代码文档</h1>
                <div class="subtitle">基于数据驱动的连锁零售药店需求预测与补货优化系统 （核心业务版）</div>
                <div class="info">
                    TFT 时序预测 + PPO 强化学习 + Optuna 超参调优<br>
                    扫描模式: 聚焦核心业务逻辑（排除缓存/虚拟环境/构建产物/IDE配置）<br>
                    生成时间: {gen_date}
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">文件总数</div>
                    <div class="value">{total_files}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Python 类</div>
                    <div class="value">{total_classes}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Python 函数/方法</div>
                    <div class="value">{total_functions}</div>
                </div>
                <div class="stat-card">
                    <div class="label">总代码行</div>
                    <div class="value">{total_lines:,}</div>
                </div>
            </div>
"""

HTML_FOOTER = """
            <div class="footer">
                DRP-Optima 代码文档 （核心业务版）· 由 generate_code_doc.py 自动生成 · {gen_date}
            </div>
        </main>
    </div>
    
    <script>
        // 折叠/展开
        function toggleSection(el) {{
            el.classList.toggle('collapsed');
        }}
        
        // 搜索过滤
        function filterContent(query) {{
            const q = query.toLowerCase().trim();
            const allFileSections = document.querySelectorAll('.file-section');
            if (!q) {{
                allFileSections.forEach(s => s.style.display = '');
                return;
            }}
            allFileSections.forEach(section => {{
                const text = section.textContent.toLowerCase();
                section.style.display = text.includes(q) ? '' : 'none';
            }});
        }}
        
        // 全部展开/折叠
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'e') {{
                e.preventDefault();
                document.querySelectorAll('.file-header.collapsed').forEach(h => h.classList.remove('collapsed'));
                document.querySelectorAll('.class-header.collapsed').forEach(h => h.classList.remove('collapsed'));
            }}
            if (e.ctrlKey && e.key === 'w') {{
                e.preventDefault();
                document.querySelectorAll('.file-header:not(.collapsed)').forEach(h => h.classList.add('collapsed'));
                document.querySelectorAll('.class-header:not(.collapsed)').forEach(h => h.classList.add('collapsed'));
            }}
        }});
    </script>
</body>
</html>
"""


# ── 主流程 ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DRP-Optima 代码文档生成器 (聚焦核心业务逻辑版 v3)")
    print("=" * 60)
    
    # 扫描所有文件
    print("\n🔍 扫描项目文件...")
    all_files = scan_all_files(PROJECT_ROOT)
    print(f"   找到 {len(all_files)} 个支持的文件")
    
    # 解析文件
    print("\n📖 解析文件...")
    all_files_data = []
    total_files = 0
    total_classes = 0
    total_functions = 0
    total_lines = 0
    
    for i, (rel_path, abs_path, file_type) in enumerate(all_files):
        # 每 10 个文件打印一次进度（更详细）
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(all_files)} ({(i+1)/len(all_files)*100:.1f}%) | 当前: {rel_path[:50]}...")
        
        # 读取文件内容
        source = read_file_safe(abs_path)
        
        file_data = {
            "path": rel_path,
            "filename": abs_path.name,
            "file_type": file_type,
            "line_count": len(source.splitlines()) if source else 0,
            "source": source,
            "is_python": file_type == "python",
        }
        
        # 如果是 Python 文件，尝试解析
        if file_type == "python":
            parsed = parse_python_file(abs_path)
            if parsed:
                all_files_data.append(parsed)
                total_files += 1
                total_classes += len(parsed["classes"])
                total_functions += len(parsed["functions"])
                for cls in parsed["classes"]:
                    total_functions += len(cls["methods"])
                total_lines += parsed["line_count"]
                continue
        
        # 非 Python 文件或解析失败
        all_files_data.append(file_data)
        total_files += 1
        total_lines += file_data["line_count"]
    
    print(f"  进度: {len(all_files)}/{len(all_files)} (100.0%)")
    
    stats = {
        "total_files": total_files,
        "total_classes": total_classes,
        "total_functions": total_functions,
        "total_lines": total_lines,
    }
    
    print(f"\n📊 统计: {total_files} 文件 / {total_classes} 类 / {total_functions} 函数 / {total_lines:,} 行")
    
    # 生成 HTML（增量写入）
    print("\n📝 生成 HTML 文档（增量写入）...")
    output_path = PROJECT_ROOT / "docs" / "DRP-Optima_代码文档_核心业务版.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_html_incremental(all_files_data, stats, output_path)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ 文档已生成: {output_path}")
    print(f"   文件大小: {size_mb:.2f} MB")
    print(f"\n💡 提示: 由于聚焦核心业务逻辑，HTML 文件已显著减小，浏览器加载更快。")


if __name__ == "__main__":
    main()
