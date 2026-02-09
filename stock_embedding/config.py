#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — 共享配置和工具函数

提供所有模块共用的：
- 股票代码格式检测与统一转换
- 因子CSV列名自动检测
- GPU设备检测
- 高效因子数据加载
"""

import os
import re
import gc
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================
# 默认参数（可通过命令行覆盖）
# ============================================================
DEFAULT_EMBED_DIM = 16
DEFAULT_WINDOW = 20
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3


# ============================================================
# GPU 设备检测
# ============================================================
def get_device():
    """自动检测可用设备，有 CUDA 用 CUDA，否则用 CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[设备] 使用 GPU: {torch.cuda.get_device_name(0)}")
            return device
    except ImportError:
        pass
    print("[设备] 使用 CPU")
    try:
        import torch
        return torch.device('cpu')
    except ImportError:
        return None


# ============================================================
# 股票代码格式检测与统一转换
# ============================================================
# 支持格式: 000001.SZ, 600000.SH, SZ000001, SH600000, 000001.XSHE, 600000.XSHG, 纯数字000001
_PATTERNS = {
    'dot_suffix': re.compile(r'^(\d{6})\.(SZ|SH|XSHE|XSHG)$', re.IGNORECASE),
    'prefix': re.compile(r'^(SZ|SH)(\d{6})$', re.IGNORECASE),
    'pure_digits': re.compile(r'^(\d{6})$'),
}

# 交易所推断规则: 6开头 → SH，其余 → SZ
def _infer_exchange(code_digits: str) -> str:
    return 'SH' if code_digits.startswith('6') else 'SZ'


def _normalize_exchange(ex: str) -> str:
    """把 XSHE/XSHG 等格式统一为 SZ/SH"""
    ex = ex.upper()
    if ex in ('SZ', 'XSHE'):
        return 'SZ'
    if ex in ('SH', 'XSHG'):
        return 'SH'
    return ex


def normalize_stock_code(code) -> str:
    """
    将各种格式的股票代码统一为 XXXXXX.SZ / XXXXXX.SH 格式
    """
    code = str(code).strip()

    # 格式1: 000001.SZ / 000001.XSHE
    m = _PATTERNS['dot_suffix'].match(code)
    if m:
        digits, ex = m.group(1), m.group(2)
        return f"{digits}.{_normalize_exchange(ex)}"

    # 格式2: SZ000001 / SH600000
    m = _PATTERNS['prefix'].match(code)
    if m:
        ex, digits = m.group(1), m.group(2)
        return f"{digits}.{_normalize_exchange(ex)}"

    # 格式3: 纯数字 000001
    m = _PATTERNS['pure_digits'].match(code)
    if m:
        digits = m.group(1)
        return f"{digits}.{_infer_exchange(digits)}"

    # 无法识别
    warnings.warn(f"[警告] 无法识别的股票代码格式: {code}，保留原值")
    return code


def detect_code_format(codes):
    """
    检测一组股票代码的格式类型
    返回: 'dot_suffix' | 'prefix' | 'pure_digits' | 'unknown'
    """
    sample = [str(c).strip() for c in list(codes)[:20]]
    for fmt, pat in _PATTERNS.items():
        if all(pat.match(c) for c in sample if c):
            return fmt
    return 'unknown'


def normalize_stock_codes_series(series: pd.Series) -> pd.Series:
    """向量化地转换整列股票代码"""
    return series.astype(str).str.strip().map(normalize_stock_code)


# ============================================================
# 列名自动检测
# ============================================================
_DATE_NAMES = {'tradingdate', 'tradingday', 'date', 'trade_date', 'tradedate',
               '日期', '交易日期', '交易日'}
_STOCK_NAMES = {'securityid', 'security_id', 'stock_code', 'stockcode', 'stock',
                'ticker', 'symbol', 'code', 'windcode', 'wind_code',
                '股票代码', '证券代码', '代码'}
_VALUE_NAMES = {'value', 'val', 'factor_value', '值', '因子值'}


def detect_column(df_columns, candidates, role_name=''):
    """
    在 df 的列名中自动检测匹配的列
    candidates: 候选列名集合（全小写）
    role_name: 用于报错提示
    返回: 匹配到的原始列名
    """
    cols_lower = {c.lower().strip(): c for c in df_columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # 模糊匹配: 列名包含关键字
    for cand in candidates:
        for cl, orig in cols_lower.items():
            if cand in cl:
                return orig
    return None


def detect_date_col(df):
    col = detect_column(df.columns, _DATE_NAMES, '日期')
    if col is None:
        raise ValueError(
            f"[错误] 无法检测日期列。当前列名: {list(df.columns)}\n"
            f"  支持的列名: {_DATE_NAMES}\n"
            f"  请重命名日期列为 'TradingDate'"
        )
    return col


def detect_stock_col(df):
    col = detect_column(df.columns, _STOCK_NAMES, '股票代码')
    if col is None:
        raise ValueError(
            f"[错误] 无法检测股票代码列。当前列名: {list(df.columns)}\n"
            f"  支持的列名: {_STOCK_NAMES}\n"
            f"  请重命名股票代码列为 'SecurityID'"
        )
    return col


def detect_value_col(df, date_col, stock_col):
    """检测值列：先按名称匹配，匹配不到则取除日期和股票列以外的第一列"""
    col = detect_column(df.columns, _VALUE_NAMES, '值')
    if col is not None:
        return col
    remaining = [c for c in df.columns if c not in (date_col, stock_col)]
    if remaining:
        return remaining[0]
    raise ValueError(
        f"[错误] 无法检测值列。当前列名: {list(df.columns)}\n"
        f"  日期列={date_col}, 股票列={stock_col}, 无剩余列可用"
    )


# ============================================================
# 日期格式统一
# ============================================================
def normalize_date_str(date_str: str) -> str:
    """将各种日期格式统一为 YYYY.MM.DD"""
    s = str(date_str).strip()
    # 尝试用 pandas 解析
    try:
        dt = pd.to_datetime(s)
        return dt.strftime('%Y.%m.%d')
    except Exception:
        return s


def parse_dates_series(series: pd.Series) -> pd.Series:
    """向量化解析日期列为 datetime"""
    s = series.astype(str).str.strip()
    # 替换点号为短横线以便 pandas 解析
    s_clean = s.str.replace('.', '-', regex=False)
    return pd.to_datetime(s_clean, errors='coerce')


# ============================================================
# 高效因子数据加载
# ============================================================
def _read_one_factor(filepath_str):
    """读取单个因子CSV，返回标准化的 DataFrame"""
    filepath = Path(filepath_str)
    try:
        df = pd.read_csv(filepath_str, low_memory=False)
    except Exception as e:
        warnings.warn(f"[警告] 读取文件失败 {filepath.name}: {e}")
        return None

    if len(df.columns) < 3:
        warnings.warn(f"[警告] 文件列数不足3: {filepath.name}, 列名={list(df.columns)}")
        return None

    # 自动检测列名
    date_col = detect_date_col(df)
    stock_col = detect_stock_col(df)
    value_col = detect_value_col(df, date_col, stock_col)

    df = df[[date_col, stock_col, value_col]].copy()
    df.columns = ['date', 'stock', 'value']

    # 解析日期
    df['date'] = parse_dates_series(df['date'])

    # 统一股票代码
    df['stock'] = normalize_stock_codes_series(df['stock'])

    # 值转为 float
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # 去重
    df = df.drop_duplicates(subset=['date', 'stock'], keep='last')
    df = df.dropna(subset=['date'])

    df['factor'] = filepath.stem
    return df


def load_factors_fast(folder, n_jobs=None):
    """
    并行读取所有因子CSV，返回:
      data_dict: {date(Timestamp): DataFrame(index=stock, columns=factor_names)}
      factor_names: 因子名列表（文件名去后缀）
      all_stocks: 所有股票代码的有序列表
      all_dates: 所有日期的有序列表
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"[错误] 因子文件夹不存在: {folder}\n  请检查路径是否正确")

    files = sorted(folder.glob('*.csv'))
    if len(files) == 0:
        raise FileNotFoundError(
            f"[错误] 因子文件夹中没有CSV文件: {folder}\n"
            f"  请确保文件夹中包含因子CSV文件"
        )

    factor_names = [f.stem for f in files]
    print(f"[数据加载] 找到 {len(files)} 个因子文件，开始并行加载...")

    n_workers = n_jobs or min(cpu_count(), 8, len(files))
    # 如果文件数少于4个，直接串行读取（避免进程启动开销）
    if len(files) <= 3:
        all_dfs = [_read_one_factor(str(f)) for f in files]
    else:
        try:
            with Pool(n_workers) as pool:
                all_dfs = pool.map(_read_one_factor, [str(f) for f in files])
        except Exception:
            # multiprocessing 失败时回退到串行
            print("[警告] 多进程加载失败，回退到串行模式")
            all_dfs = [_read_one_factor(str(f)) for f in files]

    # 过滤失败的
    all_dfs = [df for df in all_dfs if df is not None]
    if len(all_dfs) == 0:
        raise ValueError("[错误] 所有因子文件读取失败，请检查文件格式")

    combined = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    # 获取所有股票和日期
    all_stocks = sorted(combined['stock'].unique())
    all_dates = sorted(combined['date'].dropna().unique())

    print(f"[数据加载] 共 {len(all_stocks)} 只股票, {len(all_dates)} 个交易日, {len(factor_names)} 个因子")

    # 用 groupby + pivot 构建每日矩阵
    from tqdm import tqdm
    data_dict = {}
    for date, group in tqdm(combined.groupby('date'), desc='构建每日因子矩阵', leave=False):
        pivot = group.pivot_table(index='stock', columns='factor', values='value', aggfunc='last')
        pivot = pivot.reindex(columns=factor_names)
        data_dict[date] = pivot

    del combined
    gc.collect()

    return data_dict, factor_names, all_stocks, all_dates


def build_factor_tensor(data_dict, all_stocks, all_dates, factor_names, window):
    """
    从 data_dict 构建滑动窗口张量
    返回:
      X: np.ndarray, shape (n_samples, window, n_factors)
      sample_labels: list of (date, stock) 元组
    """
    from tqdm import tqdm

    n_factors = len(factor_names)
    sorted_dates = sorted(all_dates)

    samples = []
    labels = []

    for i in tqdm(range(window, len(sorted_dates)), desc='构建滑动窗口', leave=False):
        date = sorted_dates[i]
        window_dates = sorted_dates[i - window:i]

        # 获取当日有数据的股票
        if date not in data_dict:
            continue
        stocks_today = data_dict[date].index.tolist()

        for stock in stocks_today:
            seq = np.zeros((window, n_factors), dtype=np.float32)
            for t, d in enumerate(window_dates):
                if d in data_dict and stock in data_dict[d].index:
                    row = data_dict[d].loc[stock].values
                    seq[t] = np.nan_to_num(row, nan=0.0).astype(np.float32)
            samples.append(seq)
            labels.append((date, stock))

    if len(samples) == 0:
        raise ValueError(
            f"[错误] 无法构建任何样本。窗口大小={window}, 日期数={len(sorted_dates)}\n"
            f"  请确保日期数 > 窗口大小"
        )

    X = np.stack(samples, axis=0)
    return X, labels


def ensure_output_dir(path):
    """确保输出目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # 测试股票代码转换
    test_codes = ['000001.SZ', '600000.SH', 'SZ000001', 'SH600000',
                  '000001.XSHE', '600000.XSHG', '000001', '600000']
    print("股票代码转换测试:")
    for c in test_codes:
        print(f"  {c:>15s} → {normalize_stock_code(c)}")
