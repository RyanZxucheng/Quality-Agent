"""
外部检索调度器
统一接口调用多个外部搜索工具，按配置列表顺序依次执行，全部跑完后汇总返回。

已实现的工具：
  - pubmed：PubMed E-utilities API（免费，无需 API key）
  - bing_search：Bing 搜索占位工具（默认禁用）

新增工具：继承 BaseExternalTool 并注册到 TOOL_REGISTRY。
"""
import logging
import json
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# 需要在顶部导入 threading（已存在），此处仅为标记修改位置

import requests

from src.config import ExternalSearchConfig, ExternalToolConfig, get_config
from src.models import ExternalEvidence

logger = logging.getLogger(__name__)


# ── 工具基类 ─────────────────────────────────────────────────────────────────

class BaseExternalTool(ABC):
    """外部检索工具基类"""

    name: str = ""

    def __init__(self, tool_cfg: ExternalToolConfig, timeout: int = 10, max_results: int = 2):
        self.tool_cfg = tool_cfg
        self.timeout = timeout
        self.max_results = max_results

    @abstractmethod
    def search(self, query: str) -> List[ExternalEvidence]:
        """执行搜索，返回证据列表"""

    def _build_query(self, missing_slot: str) -> str:
        """根据配置的查询模板构造查询字符串"""
        return self.tool_cfg.query_template.replace("{missing_slot}", missing_slot)


# ── PubMed 工具 ───────────────────────────────────────────────────────────────

# ── PubMed 全局速率限制 ───────────────────────────────────────────────────────

_PUBMED_LOCK = threading.Lock()
_PUBMED_LAST_CALL_TIME: float = 0.0
_PUBMED_MIN_INTERVAL: float = 0.34  # NCBI 要求：无 key 时每秒最多 3 次请求


def _pubmed_rate_limit_wait() -> None:
    """线程安全的 PubMed 请求间隔控制（sleep 在锁内，确保严格串行）"""
    global _PUBMED_LAST_CALL_TIME
    with _PUBMED_LOCK:
        now = time.time()
        elapsed = now - _PUBMED_LAST_CALL_TIME
        if elapsed < _PUBMED_MIN_INTERVAL:
            time.sleep(_PUBMED_MIN_INTERVAL - elapsed)
            _PUBMED_LAST_CALL_TIME = time.time()
        else:
            _PUBMED_LAST_CALL_TIME = now


class PubMedTool(BaseExternalTool):
    """
    PubMed E-utilities 搜索工具
    文档：https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """

    name = "pubmed"
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str) -> List[ExternalEvidence]:
        built_query = self._build_query(query)
        pmids = self._esearch(built_query)
        if not pmids:
            return []

        results = []
        for pmid in pmids[: self.max_results]:
            _pubmed_rate_limit_wait()
            abstract = self._efetch_abstract(pmid)
            if abstract:
                results.append(ExternalEvidence(
                    tool_name=self.name,
                    source=f"PubMed PMID:{pmid}",
                    snippet=abstract[:500],
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    query_used=built_query,
                ))

        return results

    def _esearch(self, query: str) -> List[str]:
        """搜索 PubMed 返回 PMID 列表"""
        _pubmed_rate_limit_wait()
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.tool_cfg.api_key:
            params["api_key"] = self.tool_cfg.api_key

        try:
            resp = requests.get(self.ESEARCH_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.warning(f"PubMed esearch failed for query '{query}': {e}")
            return []

    def _efetch_abstract(self, pmid: str) -> str:
        """获取摘要文本"""
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
        }
        if self.tool_cfg.api_key:
            params["api_key"] = self.tool_cfg.api_key

        try:
            resp = requests.get(self.EFETCH_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text.strip()[:800]
        except Exception as e:
            logger.warning(f"PubMed efetch failed for PMID {pmid}: {e}")
            return ""


# ── Bing Search 工具 ──────────────────────────────────────────────────────────

class BingSearchTool(BaseExternalTool):
    """
    Bing 网页搜索（HTML 抓取，无需 API key）
    使用 cn.bing.com，适合中文查询；依赖 beautifulsoup4。
    """

    name = "bing_search"

    def search(self, query: str) -> List[ExternalEvidence]:
        try:
            from bs4 import BeautifulSoup, Tag
            from typing import cast
            from urllib.parse import urlencode
        except ImportError:
            logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            return []

        built_query = self._build_query(query)

        try:
            encoded = urlencode({"q": built_query})
            url = f"https://cn.bing.com/search?{encoded}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)

            if response.status_code != 200:
                logger.warning(f"Bing returned status code: {response.status_code}")
                return []

            response.encoding = "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")

            b_results_element = soup.find("ol", id="b_results")
            if b_results_element is None:
                logger.warning("Could not find ol#b_results element, page structure may have changed.")
                return []

            b_results_tag = cast(Tag, b_results_element)
            result_items = b_results_tag.find_all("li")

            results: List[ExternalEvidence] = []
            for i in range(min(len(result_items), self.max_results)):
                row = result_items[i]
                if not isinstance(row, Tag):
                    continue

                h2_element = row.find("h2")
                if h2_element is None:
                    continue
                h2_tag = cast(Tag, h2_element)

                title = h2_tag.get_text().strip()

                link_tag_element = h2_tag.find("a")
                if link_tag_element is None:
                    continue
                link_tag = cast(Tag, link_tag_element)

                link = link_tag.get("href")
                if link is None:
                    continue

                content_element = row.find("p", class_="b_algoSlug")
                content_text = ""
                if content_element is not None and isinstance(content_element, Tag):
                    content_text = content_element.get_text()

                results.append(
                    ExternalEvidence(
                        tool_name=self.name,
                        source=title,
                        snippet=content_text[:500],
                        url=str(link),
                        query_used=built_query,
                    )
                )

            if not results:
                logger.warning(
                    f"No parsed results for built_query={built_query!r}. "
                    "Check if Bing HTML structure has changed."
                )

            return results

        except Exception as e:
            logger.warning(f"Bing scraping error for query '{built_query}': {e!s}")
            return []




# ── Baidu Search 工具 ─────────────────────────────────────────────────────────

class BaiduSearchTool(BaseExternalTool):
    """
    百度网页搜索（HTML 抓取，无需 API key）
    依赖：pip install beautifulsoup4

    使用 Session 先访问首页取得 cookie，再发起搜索请求，避免被拦截为超时页。
    描述元素适配百度现行结构（.cu-line-clamp-2 / [class*=summary-text]）。
    """

    name = "baidu_search"
    HOME_URL = "https://www.baidu.com"
    SEARCH_URL = "https://www.baidu.com/s"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    # 描述选择器：按优先级依次尝试
    _DESC_SELECTORS = [
        ".cu-line-clamp-2",
        "[class*=summary-text]",
        "[class*=abstract]",
        ".c-abstract",
        ".c-span-last",
    ]

    def __init__(self, tool_cfg, timeout: int = 10, max_results: int = 2):
        super().__init__(tool_cfg, timeout, max_results)
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()

    def _get_session(self) -> requests.Session:
        """懒加载 Session，首次调用时访问百度首页获取 cookie（线程安全）"""
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is not None:
                return self._session
            session = requests.Session()
            try:
                session.get(self.HOME_URL, headers=self.HEADERS, timeout=self.timeout)
                logger.debug("Baidu session initialized with cookies")
            except Exception as e:
                logger.warning(f"Baidu homepage prefetch failed (will retry on search): {e}")
            self._session = session
            return self._session

    def search(self, query: str) -> List[ExternalEvidence]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            return []

        built_query = self._build_query(query)
        params = {"wd": built_query, "rn": str(self.max_results*5)}

        try:
            session = self._get_session()
            resp = session.get(
                self.SEARCH_URL,
                headers={**self.HEADERS, "Referer": self.HOME_URL},
                params=params,
                timeout=self.timeout,
            )
            resp.encoding = "utf-8"
        except Exception as e:
            logger.warning(f"Baidu request failed for query '{built_query}': {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select(".result"):
            if len(results) >= self.max_results:
                break

            title_el = item.select_one("h3 a") or item.select_one("h3")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            url = title_el.get("href", "") if title_el.name == "a" else ""

            desc = ""
            for sel in self._DESC_SELECTORS:
                desc_el = item.select_one(sel)
                if desc_el:
                    desc = desc_el.get_text(strip=True)
                    break

            if not title:
                continue

            results.append(ExternalEvidence(
                tool_name=self.name,
                source=title,
                snippet=desc[:500],
                url=url,
                query_used=built_query,
            ))

        if not results:
            logger.warning(
                "Baidu search returned no results. "
                "HTML structure may have changed or request was blocked."
            )

        return results


# ── Exa MCP 工具 ──────────────────────────────────────────────────────────────

class ExaMCPTool(BaseExternalTool):
    """
    Exa MCP 搜索工具（通过 mcporter 调用 exa.web_search_exa）

    与 Agent-Reach 方案一致：
      1) 依赖本机安装 mcporter
      2) 通过 `mcporter config add exa https://mcp.exa.ai/mcp` 注册
      3) 使用 `mcporter call 'exa.web_search_exa(...)'` 执行搜索
    """

    name = "exa_mcp"
    DEFAULT_ALIAS = "exa"
    DEFAULT_TOOL = "web_search_exa"

    def __init__(self, tool_cfg: ExternalToolConfig, timeout: int = 10, max_results: int = 2):
        super().__init__(tool_cfg, timeout, max_results)
        # 兼容 endpoint 字段：允许写成 `exa`（默认别名）
        alias = (tool_cfg.endpoint or "").strip()
        self.alias = alias if alias else self.DEFAULT_ALIAS

    def search(self, query: str) -> List[ExternalEvidence]:
        built_query = self._build_query(query)
        mcporter = shutil.which("mcporter")
        if not mcporter:
            logger.warning(
                "Exa MCP unavailable: mcporter not found. Install with `npm install -g mcporter` "
                "and run `mcporter config add exa https://mcp.exa.ai/mcp`."
            )
            return []

        if not self._has_alias(mcporter):
            logger.warning(
                "Exa MCP unavailable: alias '%s' not configured in mcporter. "
                "Run `mcporter config add %s https://mcp.exa.ai/mcp`.",
                self.alias,
                self.alias,
            )
            return []

        raw = self._call_web_search(mcporter, built_query)
        if raw is None:
            return []

        return self._parse_evidence(raw, built_query)

    def _run(self, args: List[str]) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning(f"Exa MCP command failed: args={args!r}, err={e}")
            return None

    def _has_alias(self, mcporter: str) -> bool:
        proc = self._run([mcporter, "config", "list"])
        if proc is None:
            return False
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        return self.alias.lower() in combined

    def _escape_for_mcporter_call(self, value: str) -> str:
        # 以 JSON 字符串转义保证引号/换行安全，再去掉外层双引号。
        return json.dumps(value, ensure_ascii=False)[1:-1]

    def _call_web_search(self, mcporter: str, query: str) -> Optional[str]:
        escaped_query = self._escape_for_mcporter_call(query)
        call_expr = (
            f'{self.alias}.{self.DEFAULT_TOOL}(query: "{escaped_query}", '
            f"numResults: {self.max_results})"
        )
        proc = self._run([mcporter, "call", call_expr])
        if proc is None:
            return None
        if proc.returncode != 0:
            logger.warning(
                "Exa MCP command non-zero exit. code=%s, stderr=%s",
                proc.returncode,
                (proc.stderr or "").strip()[:300],
            )
            return None

        return (proc.stdout or "").strip()

    def _parse_evidence(self, raw_output: str, query_used: str) -> List[ExternalEvidence]:
        candidates = self._extract_result_candidates(raw_output)
        evidence: List[ExternalEvidence] = []
        for item in candidates:
            title = str(item.get("title") or item.get("source") or item.get("name") or "").strip()
            snippet = str(item.get("snippet") or item.get("text") or item.get("summary") or "").strip()
            url = str(item.get("url") or item.get("link") or "").strip()

            if not title and not snippet:
                continue

            evidence.append(
                ExternalEvidence(
                    tool_name=self.name,
                    source=title or "Exa MCP result",
                    snippet=snippet[:500],
                    url=url,
                    query_used=query_used,
                )
            )
            if len(evidence) >= self.max_results:
                break
        return evidence

    def _extract_result_candidates(self, raw_output: str) -> List[Dict[str, Any]]:
        text = (raw_output or "").strip()
        if not text:
            return []

        # 优先解析 Exa 纯文本格式（Title/URL/Highlights）
        if "Title:" in text and "URL:" in text:
            return self._parse_exa_text_format(text)

        parsed_values: List[Any] = []
        try:
            parsed_values.append(json.loads(text))
        except Exception:
            pass

        # mcporter 某些版本会输出多行，逐行尝试 JSON 解析。
        if not parsed_values:
            for line in text.splitlines():
                line = line.strip()
                if not line or (not line.startswith("{") and not line.startswith("[")):
                    continue
                try:
                    parsed_values.append(json.loads(line))
                except Exception:
                    continue

        candidates: List[Dict[str, Any]] = []
        for obj in parsed_values:
            candidates.extend(self._scan_dict_for_hits(obj))

        # 最后兜底：把纯文本输出作为一条 snippet 返回
        if not candidates:
            candidates.append({"snippet": text[:1200]})
        return candidates

    def _parse_exa_text_format(self, text: str) -> List[Dict[str, Any]]:
        """解析 Exa web_search_exa 的纯文本返回格式。"""
        blocks = [b.strip() for b in text.split("---") if b.strip()]
        candidates: List[Dict[str, Any]] = []
        # Exa 结果中常见的字段头，遇到它们应退出 highlights 区域
        KNOWN_FIELD_PREFIXES = ("Title:", "URL:", "Published:", "Author:", "Highlights:")

        for block in blocks:
            title = ""
            url = ""
            highlights: List[str] = []
            in_highlights = False

            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith("Title:"):
                    title = stripped[len("Title:"):].strip()
                    in_highlights = False
                elif stripped.startswith("URL:"):
                    url = stripped[len("URL:"):].strip()
                    in_highlights = False
                elif stripped.startswith("Highlights:"):
                    in_highlights = True
                elif any(stripped.startswith(p) for p in KNOWN_FIELD_PREFIXES):
                    # 其他已知字段，退出 highlights
                    in_highlights = False
                elif in_highlights and stripped:
                    # 在 Highlights 区域内，收集所有非空行
                    if stripped.startswith("-"):
                        highlights.append(stripped[1:].strip())
                    else:
                        highlights.append(stripped)
                elif stripped == "":
                    continue

            snippet = "\n".join(highlights)
            candidates.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })
        return candidates

    def _scan_dict_for_hits(self, obj: Any) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        if isinstance(obj, dict):
            has_url = any(k in obj for k in ("url", "link"))
            has_text = any(k in obj for k in ("snippet", "text", "summary", "title", "source", "name"))
            if has_url or has_text:
                hits.append(obj)
            for value in obj.values():
                hits.extend(self._scan_dict_for_hits(value))
        elif isinstance(obj, list):
            for value in obj:
                hits.extend(self._scan_dict_for_hits(value))
        return hits


# ── 工具注册表 ────────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, type] = {
    "pubmed": PubMedTool,
    "bing_search": BingSearchTool,
    "baidu_search": BaiduSearchTool,
    "exa_mcp": ExaMCPTool,
}


# ── 调度器 ────────────────────────────────────────────────────────────────────

class ExternalSearchRunner:
    """
    外部检索调度器
    默认并行调用所有启用的工具（各工具在独立线程中运行），全部完成后按原始配置顺序合并结果。
    可通过 config.parallel_tools=False 回退为串行。
    """

    def __init__(self, config: Optional[ExternalSearchConfig] = None):
        self.config = config or get_config().external_search
        self._tools: Optional[List[BaseExternalTool]] = None
        self._warm_up_lock = threading.Lock()

    def _build_tools(self) -> List[BaseExternalTool]:
        """构建已启用的工具列表（按配置列表顺序）"""
        tools = []
        for tool_cfg in [t for t in self.config.tools if t.enabled]:
            cls = TOOL_REGISTRY.get(tool_cfg.name)
            if cls is None:
                logger.warning(f"Unknown external tool: '{tool_cfg.name}', skipping")
                continue
            tools.append(cls(tool_cfg, timeout=self.config.timeout, max_results=self.config.max_results_per_tool))
        return tools

    def warm_up(self) -> None:
        """预加载工具列表，避免并发时的懒加载竞争（线程安全）"""
        if self._tools is not None:
            return
        with self._warm_up_lock:
            if self._tools is not None:
                return
            self._tools = self._build_tools()
        logger.debug(f"ExternalSearchRunner warmed up with {len(self._tools)} tools")

    def fetch(self, missing_slots: str) -> List[ExternalEvidence]:
        """
        针对缺失信息描述执行外部检索（单次查询）

        当 config.parallel_tools=True（默认）时，所有启用工具并发执行，
        结果按原始工具配置顺序合并，任一工具失败不影响其余结果。

        Args:
            missing_slots: 自检识别出的缺失信息描述（自然语言段落）

        Returns:
            ExternalEvidence 列表（可能为空）
        """
        if not self.config.enabled:
            return []

        if not missing_slots:
            return []

        if self._tools is None:
            self._tools = self._build_tools()

        if not self._tools:
            logger.info("No external search tools enabled")
            return []

        if getattr(self.config, "parallel_tools", True) and len(self._tools) > 1:
            return self._fetch_parallel(missing_slots)
        return self._fetch_sequential(missing_slots)

    def _fetch_sequential(self, missing_slots: str) -> List[ExternalEvidence]:
        """串行依次调用各工具"""
        all_evidence: List[ExternalEvidence] = []
        for tool in self._tools:
            try:
                logger.debug(f"External search [{tool.name}] for: {missing_slots[:120]}")
                results = tool.search(missing_slots)
                if results:
                    all_evidence.extend(results)
            except Exception as e:
                logger.error(f"External tool '{tool.name}' raised unexpected error: {e}")
        return all_evidence

    def _fetch_parallel(self, missing_slots: str) -> List[ExternalEvidence]:
        """并行调用所有工具，结果按原始工具顺序合并"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tool_results: Dict[int, List[ExternalEvidence]] = {}
        futures_map: Dict[Any, int] = {}

        with ThreadPoolExecutor(max_workers=len(self._tools)) as executor:
            for idx, tool in enumerate(self._tools):
                logger.debug(f"External search [{tool.name}] (parallel) for: {missing_slots[:120]}")
                future = executor.submit(self._call_tool, tool, missing_slots)
                futures_map[future] = idx

            for future in as_completed(futures_map):
                idx = futures_map[future]
                try:
                    tool_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"External tool future for index {idx} raised: {e}")
                    tool_results[idx] = []

        all_evidence: List[ExternalEvidence] = []
        for idx in sorted(tool_results):
            all_evidence.extend(tool_results[idx])
        return all_evidence

    @staticmethod
    def _call_tool(tool: BaseExternalTool, missing_slots: str) -> List[ExternalEvidence]:
        """在线程内安全调用单个工具"""
        try:
            return tool.search(missing_slots) or []
        except Exception as e:
            logger.error(f"External tool '{tool.name}' raised unexpected error: {e}")
            return []
