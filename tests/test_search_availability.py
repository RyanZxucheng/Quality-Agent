"""
检索可用性预检测试

在运行主流程前，检测当前配置下启用的检索功能是否真正可用。

用法:
    python -m unittest tests.test_search_availability -v
"""
import os
import sys
# 添加项目根目录到 Python 路径，确保可以导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import unittest
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.config import get_config
from src.evidence.internal_search import InternalSearchExecutor
from src.tools.external_search import ExternalSearchRunner, TOOL_REGISTRY
from src.evidence.reranker import create_reranker
from src.models import QAPair

logger = logging.getLogger(__name__)


class TestSearchAvailability(unittest.TestCase):
    """检索可用性测试"""

    @classmethod
    def setUpClass(cls):
        cls.config = get_config()

    @staticmethod
    def _build_test_missing_slot(query_template: str) -> str:
        """根据工具查询模板生成测试用 missing_slot。"""
        template = (query_template or "{missing_slot}").strip()
        generated = template.replace("{missing_slot}", "").strip()
        return generated or "medical evidence"

    # -- 内部检索 -----------------------------------------------------------

    def test_internal_search(self):
        """测试内部检索是否可用"""
        cfg = self.config.internal_search
        if not cfg.enabled:
            self.skipTest("Config disabled")

        executor = InternalSearchExecutor(cfg)
        ok = executor._ensure_index()
        self.assertTrue(
            ok,
            "Internal search index not available. "
            f"Expected index at: {Path(cfg.index_dir) / 'chunks.jsonl'}"
        )

        # 执行一次简单搜索
        result = executor.search(
            question="diabetes treatment",
            answer="",
            missing_slots="diabetes treatment guidelines"
        )
        self.assertIsNotNone(result)
        # 将结果信息存储在测试实例中，供测试运行器使用
        self._test_result_info = f"返回 {len(result.chunks)} chunks"

    # -- 外部检索 -----------------------------------------------------------

    def _test_external_tool(self, tool_name: str, query: Optional[str] = None):
        """测试单个外部检索工具"""
        cfg = self.config.external_search
        if not cfg.enabled:
            self.skipTest("External search config disabled")

        # 查找该工具在配置中是否启用
        tool_cfg = None
        for t in cfg.tools:
            if t.name == tool_name and t.enabled:
                tool_cfg = t
                break

        if tool_cfg is None:
            self.skipTest(f"Tool {tool_name} config disabled")

        cls = TOOL_REGISTRY.get(tool_name)
        if cls is None:
            self.fail(f"Tool '{tool_name}' not found in registry")

        query_to_use = query or self._build_test_missing_slot(tool_cfg.query_template)
        tool = cls(tool_cfg, timeout=cfg.timeout, max_results=cfg.max_results_per_tool)
        results = tool.search(query_to_use)

        self.assertIsNotNone(
            results,
            f"Tool '{tool_name}' returned None instead of list"
        )
        # 将结果信息存储在测试实例中
        if results:
            self._test_result_info = f"返回 {len(results)} 条结果"
        else:
            self._test_result_info = "返回 0 条结果 (可能网络/查询问题)"

    def test_external_pubmed(self):
        """测试 PubMed 外部检索"""
        self._test_external_tool("pubmed")

    def test_external_bing(self):
        """测试 Bing 外部检索"""
        self._test_external_tool("bing_search")

    def test_external_baidu(self):
        """测试百度外部检索"""
        self._test_external_tool("baidu_search")

    def test_external_exa_mcp(self):
        """测试 Exa MCP 外部检索"""
        self._test_external_tool("exa_mcp")

    def test_external_runner(self):
        """测试外部检索调度器整体"""
        cfg = self.config.external_search
        if not cfg.enabled:
            self.skipTest("External search config disabled")

        runner = ExternalSearchRunner(cfg)
        runner.warm_up()

        if not runner._tools:
            self.skipTest("No external tools available")

        loaded_tool_names = {tool.name for tool in runner._tools}
        enabled_tool_cfg = next(
            (t for t in cfg.tools if t.enabled and t.name in loaded_tool_names),
            None,
        )
        if enabled_tool_cfg is None:
            self.skipTest("No enabled and loadable tools found in config")

        missing_slot = self._build_test_missing_slot(enabled_tool_cfg.query_template)
        results = runner.fetch(missing_slot)
        self.assertIsNotNone(results)
        # 将结果信息存储在测试实例中
        self._test_result_info = f"返回 {len(results)} 条总结果"

    # -- Rerank ---------------------------------------------------------------

    def test_rerank(self):
        """测试 Reranker 是否可用"""
        cfg = self.config.rerank
        if not cfg.enabled:
            self.skipTest("Config disabled")

        configured_top_n = max(1, int(cfg.top_n))

        try:
            reranker = create_reranker(cfg)
        except Exception as e:
            self.fail(f"Failed to create reranker: {e}")

        if reranker is None:
            self.fail("Reranker creation returned None -- check backend config")

        # 构造候选数据做简单测试
        from src.models import RankedResult
        candidates = [
            RankedResult(
                source="internal" if i % 2 == 0 else "external",
                content=f"test candidate {i} for rerank",
                relevance_score=0.0,
            )
            for i in range(configured_top_n + 1)
        ]

        try:
            ranked = reranker.rerank(
                "diabetes treatment",
                candidates,
                top_n=configured_top_n,
            )
            self.assertIsNotNone(ranked)
            self.assertLessEqual(
                len(ranked),
                configured_top_n,
                f"Reranker returned {len(ranked)} results, expected <= {configured_top_n}",
            )
            # 将结果信息存储在测试实例中
            backend_info = f" ({cfg.backend})" if hasattr(cfg, 'backend') else ""
            self._test_result_info = (
                f"返回 {len(ranked)} 条结果{backend_info} (top_n={configured_top_n})"
            )
        except Exception as e:
            self.fail(f"Reranker execution failed: {e}")


# =============================================================================
# 美观测试运行器
# =============================================================================

def _supports_color() -> bool:
    """检查终端是否支持颜色"""
    # 尝试导入 colorama
    try:
        import colorama
        colorama.init()
        return True
    except ImportError:
        pass

    # 检查是否在交互式终端中
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False

    # Windows 终端检查
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            return kernel32.GetConsoleMode(kernel32.GetStdHandle(-11)) & 0x0004
        except:
            return False

    return True


# ANSI 颜色代码
class Colors:
    """ANSI 颜色代码"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # 前景色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


# 符号定义
class Symbols:
    """测试状态符号"""
    PASS = '[PASS]'
    FAIL = '[FAIL]'
    SKIP = '[SKIP]'
    RUNNING = '[RUN]'
    WARNING = '[WARN]'


# 错误详情打印机
class ErrorDetailsPrinter:
    """错误详情打印机 - 负责收集和格式化测试失败的详细错误信息"""

    def __init__(self, runner):
        """初始化错误详情打印机

        Args:
            runner: BeautifulTestRunner 实例，用于获取颜色样式等配置
        """
        self.runner = runner
        self.failure_details = []  # 存储失败详情 [(test, err, type)]
        self.error_details = []    # 存储错误详情 [(test, err, type)]

    def collect(self, results):
        """从测试结果中收集错误信息

        Args:
            results: unittest.TestResult 对象
        """
        # 收集失败信息
        for test, err in results.failures:
            self.failure_details.append((test, err, "FAILURE"))

        # 收集错误信息
        for test, err in results.errors:
            self.error_details.append((test, err, "ERROR"))

    def has_errors(self):
        """检查是否有错误或失败"""
        return len(self.failure_details) > 0 or len(self.error_details) > 0

    def print(self):
        """打印错误详情区域"""
        if not self.has_errors():
            return

        # 打印错误详情标题
        if self.runner.supports_color:
            print(f"\n{self.runner.styles['title']}[ERROR DETAILS]{self.runner.styles['reset']}")
            print(f"{self.runner.styles['dim']}{'-' * 50}{self.runner.styles['reset']}")
        else:
            print("\n[ERROR DETAILS]")
            print("-" * 50)

        # 打印所有失败
        all_details = self.failure_details + self.error_details
        for idx, (test, err, error_type) in enumerate(all_details, 1):
            self._print_single_error(idx, test, err, error_type)

    def _print_single_error(self, index: int, test, err, error_type: str):
        """打印单个错误的详细信息

        Args:
            index: 错误序号
            test: 测试实例
            err: 错误详情，可能是 traceback 字符串或异常三元组
            error_type: 错误类型 ("FAILURE" 或 "ERROR")
        """
        # 格式化测试名称
        test_name = self.runner._format_test_name(test)
        category = self.runner._get_test_category(test)

        # 根据分类添加前缀（与测试输出保持一致）
        if category == 'internal':
            prefix = '[Internal] '
        elif category == 'external_tool':
            if 'PubMed' in test_name:
                prefix = '[PubMed]   '
            elif 'Bing' in test_name:
                prefix = '[Bing]     '
            elif '百度' in test_name:
                prefix = '[Baidu]    '
            elif 'Exa' in test_name:
                prefix = '[Exa MCP]  '
            else:
                prefix = '[External] '
        elif category == 'external_runner':
            prefix = '[ExtRunner]'
        elif category == 'rerank':
            prefix = '[Reranker] '
        else:
            prefix = '[Other]    '

        # 打印错误标题
        if self.runner.supports_color:
            error_color = self.runner.styles['fail']
            test_color = self.runner.styles['warning']
            reset = self.runner.styles['reset']
            dim = self.runner.styles['dim']

            print(f"\n{prefix} {error_color}[{error_type}]{reset} {test_color}{test_name}{reset}")
            print(f"{dim}{'─' * 60}{reset}")
        else:
            print(f"\n{prefix} [{error_type}] {test_name}")
            print("-" * 60)

        # 提取并格式化堆栈跟踪。
        # unittest.TextTestResult 中 failures/errors 默认保存的是字符串 traceback。
        import traceback
        if isinstance(err, tuple) and len(err) >= 3:
            exc_type, exc_value, exc_traceback = err[:3]
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        elif isinstance(err, str):
            tb_lines = err.splitlines()
        else:
            tb_lines = [str(err)]

        # 打印完整的堆栈跟踪
        for line in tb_lines:
            # 清理行尾空白
            line = line.rstrip()
            if self.runner.supports_color:
                # 对堆栈跟踪行使用较暗的颜色
                print(f"{dim}{line}{reset}")
            else:
                print(line)


class BeautifulTestRunner(unittest.TextTestRunner):
    """美观的测试运行器"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('verbosity', 2)
        super().__init__(*args, **kwargs)
        self.supports_color = _supports_color()
        self.test_results: List[Dict] = []
        self.start_time = 0

        # 初始化错误详情打印机
        self.error_printer = ErrorDetailsPrinter(self)

        # 颜色启用时的样式
        if self.supports_color:
            self.styles = {
                'title': Colors.CYAN + Colors.BOLD,
                'section': Colors.BLUE + Colors.BOLD,
                'pass': Colors.GREEN,
                'fail': Colors.RED,
                'skip': Colors.YELLOW,
                'running': Colors.BLUE,
                'warning': Colors.MAGENTA,
                'reset': Colors.RESET,
                'dim': '\033[2m',
                'underline': '\033[4m',
            }
        else:
            self.styles = {k: '' for k in [
                'title', 'section', 'pass', 'fail', 'skip',
                'running', 'warning', 'reset', 'dim', 'underline'
            ]}

    class BeautifulTestResult(unittest.TextTestResult):
        """美观的测试结果处理器"""

        def __init__(self, stream, descriptions, verbosity, runner):
            super().__init__(stream, descriptions, verbosity)
            self.runner = runner
            self.test_times = {}

        def startTest(self, test):
            """测试开始"""
            self.test_times[test] = time.time()
            super().startTest(test)

        def _get_test_message(self, test):
            """从测试实例获取消息"""
            # 检查是否有 _test_result_info 属性
            if hasattr(test, '_test_result_info'):
                return getattr(test, '_test_result_info', '')
            return ''

        def addSuccess(self, test):
            """测试通过"""
            elapsed = time.time() - self.test_times.get(test, time.time())
            message = self._get_test_message(test)
            self.runner._print_test_result(test, 'pass', message, elapsed)
            super().addSuccess(test)

        def addFailure(self, test, err):
            """测试失败"""
            elapsed = time.time() - self.test_times.get(test, time.time())
            self.runner._print_test_result(test, 'fail', '', elapsed)
            super().addFailure(test, err)

        def addError(self, test, err):
            """测试错误"""
            elapsed = time.time() - self.test_times.get(test, time.time())
            self.runner._print_test_result(test, 'error', '', elapsed)
            super().addError(test, err)

        def addSkip(self, test, reason):
            """测试跳过"""
            elapsed = time.time() - self.test_times.get(test, time.time())
            self.runner._print_test_result(test, 'skip', reason, elapsed)
            super().addSkip(test, reason)

    def _format_test_name(self, test) -> str:
        """格式化测试名称为可读格式"""
        test_id = test.id()
        # 提取方法名（去掉类名和模块路径）
        if '.' in test_id:
            method_name = test_id.split('.')[-1]
            # 将 test_ 前缀转为更友好的名称
            if method_name.startswith('test_'):
                name = method_name[5:].replace('_', ' ')
                # 特殊处理一些已知测试
                if name == 'internal search':
                    return 'Internal Search'
                elif name == 'external runner':
                    return 'External Runner'
                elif name == 'rerank':
                    return 'Reranker'
                elif name == 'external pubmed':
                    return 'PubMed'
                elif name == 'external bing':
                    return 'Bing Search'
                elif name == 'external baidu':
                    return 'Baidu Search'
                elif name == 'external exa mcp':
                    return 'Exa MCP'
                else:
                    # 转换为标题形式
                    return name.title()
            return method_name
        return test_id

    def _get_test_category(self, test) -> str:
        """获取测试分类"""
        test_id = test.id()
        if 'internal_search' in test_id:
            return 'internal'
        elif 'external_' in test_id and 'runner' not in test_id:
            return 'external_tool'
        elif 'external_runner' in test_id:
            return 'external_runner'
        elif 'rerank' in test_id:
            return 'rerank'
        return 'other'

    def _print_header(self):
        """打印测试头信息"""
        if self.supports_color:
            print(f"\n{self.styles['title']}[TEST] Search Availability Check - Starting{self.styles['reset']}")
            print(f"{self.styles['dim']}{'-' * 50}{self.styles['reset']}\n")
        else:
            print("\n[TEST] Search Availability Check - Starting")
            print("-" * 50 + "\n")

    def _print_test_result(self, test, status: str, message: str = "", elapsed: float = 0):
        """打印单个测试结果"""
        test_name = self._format_test_name(test)
        category = self._get_test_category(test)

        # 根据分类添加前缀
        if category == 'internal':
            prefix = '[Internal] '
        elif category == 'external_tool':
            # 从测试名提取工具名
            if 'PubMed' in test_name:
                prefix = '[PubMed]   '
            elif 'Bing' in test_name:
                prefix = '[Bing]     '
            elif '百度' in test_name:
                prefix = '[Baidu]    '
            elif 'Exa' in test_name:
                prefix = '[Exa MCP]  '
            else:
                prefix = '[External] '
        elif category == 'external_runner':
            prefix = '[ExtRunner]'
        elif category == 'rerank':
            prefix = '[Reranker] '
        else:
            prefix = '[Other]    '

        # 状态符号和颜色
        if status == 'pass':
            symbol = Symbols.PASS
            color = self.styles['pass']
            status_text = '通过'
        elif status == 'fail':
            symbol = Symbols.FAIL
            color = self.styles['fail']
            status_text = '失败'
        elif status == 'skip':
            symbol = Symbols.SKIP
            color = self.styles['skip']
            status_text = '跳过'
        elif status == 'error':
            symbol = Symbols.FAIL
            color = self.styles['fail']
            status_text = '错误'
        else:
            symbol = '?'
            color = ''
            status_text = '未知'

        # 构建输出行
        time_str = f" ({elapsed:.2f}s)" if elapsed > 0 else ""
        base_line = f"{prefix} {color}{symbol}{self.styles['reset']} {test_name}"

        if message:
            if self.supports_color:
                print(f"{base_line} {self.styles['dim']}{message}{time_str}{self.styles['reset']}")
            else:
                print(f"{base_line} {message}{time_str}")
        else:
            print(f"{base_line}{time_str}")

    def _print_summary(self, results):
        """打印测试总结"""
        total = results.testsRun
        failures = len(results.failures) + len(results.errors)
        skips = len(results.skipped)
        passes = total - failures - skips
        elapsed = time.time() - self.start_time

        if self.supports_color:
            print(f"\n{self.styles['dim']}{'-' * 50}{self.styles['reset']}")
            print(f"{self.styles['title']}[SUMMARY] Test Summary{self.styles['reset']}")
            print(f"{self.styles['pass']}├── Passed: {passes}{self.styles['reset']}")
            print(f"{self.styles['skip']}├── Skipped: {skips}{self.styles['reset']}")
            print(f"{self.styles['fail']}└── Failed: {failures}{self.styles['reset']}")
            print(f"{self.styles['dim']}   Time: {elapsed:.2f}s{self.styles['reset']}")

            # 显示配置状态
            print(f"\n{self.styles['title']}[CONFIG] Config Status{self.styles['reset']}")
            try:
                config = get_config()
                internal_enabled = config.internal_search.enabled
                external_enabled = config.external_search.enabled
                rerank_enabled = config.rerank.enabled

                internal_status = f"{self.styles['pass']}[PASS]{self.styles['reset']}" if internal_enabled else f"{self.styles['skip']}[SKIP]{self.styles['reset']}"
                external_status = f"{self.styles['pass']}[PASS]{self.styles['reset']}" if external_enabled else f"{self.styles['skip']}[SKIP]{self.styles['reset']}"
                rerank_status = f"{self.styles['pass']}[PASS]{self.styles['reset']}" if rerank_enabled else f"{self.styles['skip']}[SKIP]{self.styles['reset']}"

                print(f"{internal_status} Internal Search")
                print(f"{external_status} External Search")
                print(f"{rerank_status} Reranker")

            except Exception as e:
                print(f"{self.styles['warning']}[WARN]  Cannot read config: {e}{self.styles['reset']}")

        else:
            print("\n" + "-" * 50)
            print("[SUMMARY] Test Summary")
            print(f"├── Passed: {passes}")
            print(f"├── Skipped: {skips}")
            print(f"└── Failed: {failures}")
            print(f"   Time: {elapsed:.2f}s")

    def run(self, test):
        """运行测试套件"""
        self.start_time = time.time()
        self._print_header()

        # 创建自定义结果处理器
        result = self._makeResult()
        result.failfast = self.failfast
        result.buffer = self.buffer

        # 注册结果处理器以接收输出
        if hasattr(self, '_makeBuffer'):
            # Python 3.11+ 支持缓冲输出
            result._original_stdout = sys.stdout
            result._original_stderr = sys.stderr
            sys.stdout = result._stdout_buffer = self._makeBuffer()
            sys.stderr = result._stderr_buffer = self._makeBuffer()

        # 运行测试
        test(result)

        # 恢复输出
        if hasattr(self, '_makeBuffer'):
            sys.stdout = result._original_stdout
            sys.stderr = result._original_stderr

        # 打印总结
        self._print_summary(result)

        # 收集并打印错误详情
        self.error_printer.collect(result)
        self.error_printer.print()

        return result

    def _makeResult(self):
        """创建自定义测试结果处理器"""
        return self.BeautifulTestResult(self.stream, self.descriptions, self.verbosity, self)


if __name__ == "__main__":
    # 添加项目根目录到 Python 路径，确保可以导入 src 模块
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 配置日志（只显示 WARNING 及以上，避免干扰测试输出）
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # 创建测试加载器
    loader = unittest.TestLoader()

    # 加载测试
    suite = loader.loadTestsFromTestCase(TestSearchAvailability)

    # 创建并运行美观测试运行器
    runner = BeautifulTestRunner(verbosity=0)  # verbosity=0 禁用默认输出
    result = runner.run(suite)

    # 根据测试结果退出
    sys.exit(0 if result.wasSuccessful() else 1)
