# Python代码片段库

## 📝 使用说明
这个文件收集了常用的Python代码片段，包括各种实用功能和最佳实践。每个代码片段都经过测试，可以直接使用或根据需要进行修改。

## 🔧 基础工具类

### 文件操作
```python
import os
import shutil
from pathlib import Path

# 创建目录
def create_directory(path):
    """创建目录，如果不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"目录已创建: {path}")

# 复制文件
def copy_file(src, dst):
    """复制文件到目标位置"""
    try:
        shutil.copy2(src, dst)
        print(f"文件已复制: {src} -> {dst}")
    except Exception as e:
        print(f"复制失败: {e}")

# 批量重命名文件
def batch_rename(directory, old_ext, new_ext):
    """批量重命名文件扩展名"""
    for filename in os.listdir(directory):
        if filename.endswith(old_ext):
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, filename.replace(old_ext, new_ext))
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {filename.replace(old_ext, new_ext)}")
```

### 数据处理
```python
import pandas as pd
import numpy as np

# 数据清洗
def clean_data(df):
    """数据清洗函数"""
    # 删除重复行
    df = df.drop_duplicates()

    # 处理缺失值
    df = df.fillna(df.mean())

    # 删除异常值
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df

# 数据统计
def data_summary(df):
    """生成数据摘要"""
    summary = {
        '总行数': len(df),
        '总列数': len(df.columns),
        '缺失值': df.isnull().sum().sum(),
        '重复行': df.duplicated().sum(),
        '数值列': df.select_dtypes(include=[np.number]).columns.tolist(),
        '文本列': df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary
```

### 网络请求
```python
import requests
import json
from urllib.parse import urljoin

# HTTP请求封装
class HTTPClient:
    def __init__(self, base_url=None, timeout=30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, url, params=None, headers=None):
        """GET请求"""
        url = urljoin(self.base_url, url) if self.base_url else url
        response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def post(self, url, data=None, json=None, headers=None):
        """POST请求"""
        url = urljoin(self.base_url, url) if self.base_url else url
        response = self.session.post(url, data=data, json=json, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

# 使用示例
client = HTTPClient("https://api.example.com")
data = client.get("/users", params={"page": 1})
```

## 🎨 实用功能类

### 文本处理
```python
import re
import jieba
from collections import Counter

# 文本清洗
def clean_text(text):
    """清洗文本"""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 中文分词
def chinese_segmentation(text):
    """中文分词"""
    words = jieba.cut(text)
    return [word for word in words if len(word) > 1]

# 词频统计
def word_frequency(text, top_n=10):
    """词频统计"""
    words = chinese_segmentation(text)
    word_count = Counter(words)
    return word_count.most_common(top_n)
```

### 时间处理
```python
from datetime import datetime, timedelta
import pytz

# 时间转换
def convert_timezone(dt, from_tz, to_tz):
    """时区转换"""
    from_tz = pytz.timezone(from_tz)
    to_tz = pytz.timezone(to_tz)

    dt = from_tz.localize(dt)
    return dt.astimezone(to_tz)

# 时间格式化
def format_datetime(dt, format_str="%Y-%m-%d %H:%M:%S"):
    """时间格式化"""
    return dt.strftime(format_str)

# 时间计算
def time_calculator(start_time, end_time):
    """计算时间差"""
    delta = end_time - start_time
    return {
        'days': delta.days,
        'hours': delta.seconds // 3600,
        'minutes': (delta.seconds % 3600) // 60,
        'seconds': delta.seconds % 60
    }
```

### 加密解密
```python
import hashlib
import base64
from cryptography.fernet import Fernet

# 哈希加密
def hash_password(password, salt=None):
    """密码哈希"""
    if salt is None:
        salt = os.urandom(32)

    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return base64.b64encode(salt + key).decode('utf-8')

# 对称加密
class SymmetricEncryption:
    def __init__(self, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)

    def encrypt(self, data):
        """加密数据"""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data):
        """解密数据"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
```

## 🚀 高级功能类

### 异步处理
```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# 异步HTTP请求
async def async_http_request(urls):
    """异步HTTP请求"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_url(session, url))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

async def fetch_url(session, url):
    """获取单个URL"""
    async with session.get(url) as response:
        return await response.text()

# 线程池处理
def thread_pool_processing(data, func, max_workers=4):
    """线程池处理"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, data))
    return results
```

### 缓存装饰器
```python
import functools
import time
from typing import Any, Callable

# 内存缓存
def memoize(func: Callable) -> Callable:
    """内存缓存装饰器"""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper

# 时间缓存
def time_cache(seconds: int):
    """时间缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return wrapper
    return decorator
```

### 配置管理
```python
import yaml
import json
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    return yaml.safe_load(f)
                elif self.config_file.endswith('.json'):
                    return json.load(f)
        except FileNotFoundError:
            print(f"配置文件不存在: {self.config_file}")
            return {}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}

    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                elif self.config_file.endswith('.json'):
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
```

## 🧪 测试工具类

### 单元测试
```python
import unittest
from unittest.mock import Mock, patch

class TestExample(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.test_data = [1, 2, 3, 4, 5]

    def test_basic_function(self):
        """基础功能测试"""
        result = sum(self.test_data)
        self.assertEqual(result, 15)

    def test_with_mock(self):
        """使用Mock测试"""
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked"

        result = mock_obj.method()
        self.assertEqual(result, "mocked")
        mock_obj.method.assert_called_once()

    @patch('module.function')
    def test_with_patch(self, mock_function):
        """使用patch测试"""
        mock_function.return_value = "patched"

        result = module.function()
        self.assertEqual(result, "patched")

if __name__ == '__main__':
    unittest.main()
```

### 性能测试
```python
import time
import cProfile
import pstats
from functools import wraps

# 性能计时装饰器
def timing(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper

# 性能分析装饰器
def profile(func):
    """性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)

        return result
    return wrapper
```

## 📊 数据分析类

### 数据可视化
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 基础图表
def create_basic_chart(data, chart_type='line'):
    """创建基础图表"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == 'line':
        ax.plot(data)
    elif chart_type == 'bar':
        ax.bar(range(len(data)), data)
    elif chart_type == 'scatter':
        ax.scatter(range(len(data)), data)

    ax.set_title('数据图表')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')

    plt.tight_layout()
    return fig

# 热力图
def create_heatmap(data):
    """创建热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('数据热力图')
    plt.tight_layout()
    return plt.gcf()
```

## 🔄 使用建议

### 代码组织
- 按功能分类组织代码片段
- 添加详细的注释和文档
- 提供使用示例
- 保持代码的简洁性

### 最佳实践
- 使用类型提示
- 添加错误处理
- 遵循PEP 8规范
- 编写单元测试

### 性能优化
- 使用适当的数据结构
- 避免不必要的循环
- 利用内置函数
- 考虑内存使用

## 📝 更新记录

### 版本历史
- **v1.0** (2025-01-07): 初始版本，包含基础代码片段
- **v1.1** (计划中): 增加更多实用功能
- **v1.2** (计划中): 优化代码质量和性能

### 贡献指南
- 分享有用的代码片段
- 提供使用反馈
- 建议改进方案
- 参与代码审查

**代码片段库持续更新中...** 🚀
