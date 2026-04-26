"""
文件工具模块
包含安全的文件读写操作，避免TOCTOU反模式
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def safe_read_json(file_path: str) -> List[Dict[str, Any]]:
    """
    安全读取JSON文件，直接尝试打开而不预先检查存在性

    Args:
        file_path: 文件路径

    Returns:
        解析后的数据列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON格式无效
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 支持两种格式：对象列表 或 {results: [...]}
        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("Invalid JSON format: expected list or {results: [...]}")

        return items
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in {file_path}: {e}")
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def safe_read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    安全读取JSONL文件

    Args:
        file_path: 文件路径

    Returns:
        数据列表
    """
    items = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping JSONL line {i+1} parse error: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    return items


def safe_read_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    安全读取CSV文件

    Args:
        file_path: 文件路径

    Returns:
        数据列表
    """
    items = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(row)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    return items


def ensure_dir(dir_path: str) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        dir_path: 目录路径

    Returns:
        Path对象
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(data: List[Dict], file_path: str) -> None:
    """
    写入JSONL文件

    Args:
        data: 数据列表
        file_path: 输出文件路径
    """
    ensure_dir(Path(file_path).parent)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")