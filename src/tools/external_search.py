"""
外部检索调度器（Round 3）
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
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
            abstract = self._efetch_abstract(pmid)
            if abstract:
                results.append(ExternalEvidence(
                    tool_name=self.name,
                    source=f"PubMed PMID:{pmid}",
                    snippet=abstract[:500],
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    query_used=built_query,
                ))
            time.sleep(0.34)  # NCBI 要求：无 key 时每秒最多 3 次请求

        return results

    def _esearch(self, query: str) -> List[str]:
        """搜索 PubMed 返回 PMID 列表"""
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
                logger.warning("没有找到 ol#b_results，可能是页面结构变了。")
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

    def _get_session(self) -> requests.Session:
        """懒加载 Session，首次调用时访问百度首页获取 cookie"""
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
    按配置列表顺序依次调用启用的工具，全部执行后汇总返回
    """

    def __init__(self, config: Optional[ExternalSearchConfig] = None):
        self.config = config or get_config().external_search
        self._tools: Optional[List[BaseExternalTool]] = None

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

    def fetch(self, missing_slots: str) -> List[ExternalEvidence]:
        """
        针对缺失信息描述执行外部检索（单次查询）

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

        all_evidence: List[ExternalEvidence] = []

        for tool in self._tools:
            try:
                logger.info(f"External search [{tool.name}] for: {missing_slots[:120]}")
                results = tool.search(missing_slots)
                if results:
                    all_evidence.extend(results)
            except Exception as e:
                logger.error(f"External tool '{tool.name}' raised unexpected error: {e}")

        return all_evidence
