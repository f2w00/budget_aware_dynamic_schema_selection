# Extraction

用于实验字段选择、预算控制和 SchemaRAG 评估的 Python 脚本与测试。

## 安装

项目要求 Python 3.12 或更高版本。使用 `uv` 安装依赖：

```bash
uv sync
```

也可以根据 `pyproject.toml` 使用其他 Python 环境管理工具安装依赖。

## 运行测试

```bash
python -m unittest discover -s tests
```

## 主要脚本

- `budget_selector.py`：按预算选择目标字段。
- `unbudgeted_selector.py`：执行无预算字段选择。
- `external_schemarag_selector.py`：运行外部 SchemaRAG 字段选择。
- `train_field_selector.py`：训练字段存在性和输出长度预测模型。
- `benchmarks/`：基准测试和结果分析脚本。

训练数据、模型文件、实验原始输出和论文材料不随仓库提供。运行相关脚本前，需按脚本参数准备本地输入文件和模型目录。
