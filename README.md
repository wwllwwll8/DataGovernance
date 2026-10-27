<p align="center">
  <img alt="Throughline" src="https://img.shields.io/badge/Throughline-列的主线-111111?style=for-the-badge&logo=gitbook&logoColor=white" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" /></a>
  <a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-dataframe-150458?style=flat-square&logo=pandas&logoColor=white" /></a>
  <a href="https://docs.pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/pytest-offline-0A9EDC?style=flat-square&logo=pytest&logoColor=white" /></a>
  <img alt="no API key" src="https://img.shields.io/badge/API%20key-不需要-2EA44F?style=flat-square&logo=shieldsdotio&logoColor=white" />
</p>

<h1 align="center">🧵 Throughline</h1>

<p align="center">
  在任何一次模型调用之前，先把每一列的读写主线画出来。
</p>

<p align="center">
  <img alt="pipeline" src="https://img.shields.io/badge/①_Load-JSONL-4C6EF5?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="generate" src="https://img.shields.io/badge/②_Generate-文本生成-7950F2?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="judge" src="https://img.shields.io/badge/③_Judge-UltraFeedback-F59F00?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="dedupe" src="https://img.shields.io/badge/④_Dedupe-哈希去重-12B886?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="verify" src="https://img.shields.io/badge/⑤_Verify-数学核对-E03131?style=flat-square" />
</p>

**Throughline** 在同一个进程里准备数据。算子读一步 JSONL、写下一步，并用 `input_*` / `output_*` 声明读写的列。`Pipeline.compile()` 在调用模型之前把这些声明收成静态图。LLM 任务配方覆盖文本生成和 UltraFeedback 这类偏好评判，结果可以导出成数据集工件。

v0 是单进程原型：JSONL 步骤文件、编译期列检查、文本血缘、两份任务配方、以及不调用模型的客观校验。测试和示例都走 `MockLLM`，不需要网络，也不需要 API key。

名字来自戏剧里的 **through-line**：贯穿全剧的那根线。在这里，这根线是每一列——谁写它，谁读它。`compile()` 在第一次模型调用之前把线画完。

## ✨ 它解决什么

拼写错一列名，往往要等整批生成跑完才发现。Throughline 把这件事提前到编译：

| | |
| --- | --- |
| 🧭 | `compile()` 只记录 `input_*` / `output_*`，不调用 `operator.run()`，也不调用 LLM |
| 🧵 | `draw()` 打印生产者 → 消费者，并标出没有任何后续读者的 **dead columns** |
| 🧪 | 哈希去重、数学核对、只读 SQL 不经过模型 |
| 🔌 | 真实模型接在 `LLMClient` 上；默认路径是可回放的 `MockLLM` |

## 📦 安装与运行

需要 Python 3.11+。在仓库根目录：

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 examples/pref_then_verify.py
```

`pandas` 是表类型。`pytest` 跑测试。`math-verify` 是可选额外依赖，默认路径不会 import 它。

```bash
python3 -m pip install -e ".[math]"   # 可选
```

## 🗺️ 目录

| 路径 | 作用 |
| --- | --- |
| `throughline/core/operator.py` | `OperatorABC.run(storage, ...)` 与注册表 |
| `throughline/core/storage.py` | `FileStorage` 的 JSONL 步骤链 |
| `throughline/core/pipeline.py` | `compile` / `forward` / `run`、`AutoOP`、`KeyNode`、`draw` |
| `throughline/llm.py` | `LLMClient` 协议与 `MockLLM` |
| `throughline/ops/load.py` | `LoadJSONL` |
| `throughline/ops/tasks.py` | 文本生成与 UltraFeedback 风格打分 |
| `throughline/ops/verify.py` | 哈希去重、数学/精确核对、只读 SQL |
| `examples/pref_then_verify.py` | 离线示例 |
| `tests/` | `pytest`，无网络、无密钥 |

## 🧩 算子约定

继承 `Pipeline`，把唯一的 `FileStorage` 和各个算子放在 `self` 上，`forward` 写成一串 `run`。算子必须是流水线的属性，编译才能看见它们。

```python
from throughline import FileStorage, LoadJSONL, Pipeline, TextGenerationOp

class PrefThenVerify(Pipeline):
    def __init__(self, cache_path, llm):
        super().__init__()
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL("examples/data/seeds.jsonl")
        self.gen = TextGenerationOp(llm)

    def forward(self):
        self.load.run(self.storage.step())
        self.gen.run(
            self.storage.step(),
            input_key="instruction",
            output_key="generation",
        )
```

`storage.step()` 维护一个计数器。它从 `-1` 起步，每调用一次加一，并把这一步的视图交给算子：

- 第 0 步读 `first_entry_file_name`（入口 JSONL）。入口为空时得到空表，这正是 `LoadJSONL` 作为数据源时需要的。
- `write()` 把第 `operator_step + 1` 步写成缓存目录里的 `step1.jsonl`、`step2.jsonl`……
- 下一个算子的视图读这个文件。

列参数以 `input_` 或 `output_` 开头。字符串是一列；字符串列表（`input_keys=["instruction", "generation"]`）是多列。其余参数，例如 `drop_failures=True`，是运行时选项。

## 🔍 编译：先于任何 LLM 调用

`pipeline.compile()` 也会被 `run()` 和 `draw()` 调用：

1. 把每个算子属性包进 `AutoOP`。
2. 调用你的 `forward()`。`AutoOP.run` 只记录存储视图和列参数，不调用真正的 `Operator.run`，也不调用 LLM。
3. 对照入口文件或更早算子的输出，检查每一个输入列。
4. 为每一列建立 `KeyNode`：从生产者连到读取它的算子。
5. 用记录下来的调用替换 `forward`，之后的执行才是真跑。

列名打错会抛出 `KeyError`，并写明算子、参数、以及当时存在的列：

```text
Key Matching Error in following Operators during pipeline.compile():
- Input key 'instructon' in `gen` (class <TextGenerationOp>) ...
  Check parameter 'input_key' in `gen.run()`.
Compile stopped before any operator.run() or LLM call.
```

`LoadJSONL` 是数据源。编译时从文件读出列名（本地 IO，不是 LLM），并且它必须是第一步。

## 🧵 列血缘

`pipeline.draw()` 返回一段文本。`KeyNode.readers` 就是生产者到消费者的边。一列在没有任何后续算子读取时记为 **dead**。终态产物（评判分数、校验标记）也会出现在这里：它们留在导出行上，图的其余部分不再读它们。入口里没人读的列、写在中途却被忘掉的暂存列，走的是同一份名单，拼写错误和漏接字段因此可见。

示例打印出来是这样：

```text
pipeline: load:LoadJSONL -> gen:TextGenerationOp -> judge:UltraFeedbackOp -> dedup:HashDeduplicateFilter -> math:MathVerifyFilter
column lineage
instruction: load:LoadJSONL -> gen:TextGenerationOp, judge:UltraFeedbackOp, dedup:HashDeduplicateFilter
answer: load:LoadJSONL -> math:MathVerifyFilter
generation: gen:TextGenerationOp -> judge:UltraFeedbackOp, dedup:HashDeduplicateFilter, math:MathVerifyFilter
ratings: judge:UltraFeedbackOp -> (none)
rationales: judge:UltraFeedbackOp -> (none)
overall: judge:UltraFeedbackOp -> (none)
math_correct: math:MathVerifyFilter -> (none)
dead columns: ratings, rationales, overall, math_correct
```

v0 的血缘停在这段文本。

## 🤖 任务配方与 LLM 客户端

`LLMClient.generate(chats)` 接收一批对话（`{"role", "content"}` 消息），按顺序为每段对话返回一个字符串。`MockLLM` 回放一段脚本或一个 Python 函数，并记下每一次调用。没有网络。

`TextGenerationOp` 读 `input_key`（默认 `instruction`），写 `output_key`（默认 `generation`）。

`UltraFeedbackOp` 读一条 instruction 和一条 generation。它让客户端按 UltraFeedback 的四个方面打 JSON 分：helpfulness、honesty、instruction-following、truthfulness，并写出：

- `ratings` — 方面 → 分数
- `rationales` — 方面 → 文字
- `overall` — 评判给出的总分，或各方面分数的平均

解析器也接受另一种形状：一次提示里给 **一组** generation 打分，返回平行的 `ratings` / `rationales` 列表以及 `model_name`，方面可以是 `overall-rating`。浮点数列表会取平均写入 `overall`，给以后的适配器留位置。算子本身仍然每个样本只发送一条 generation。

### 可选的任务对象

任何带有 `inputs`、`outputs`、`format_input(record) -> chat`、`format_output(text, record) -> dict` 的对象都可以传入。`TextGenerationOp(llm, task=那个对象)` 使用这些列。内置配方仍由 `LLMClient` 驱动。

## ✅ 客观校验

这些算子不调用 LLM。

- `HashDeduplicateFilter` 对 `input_key` 或 `input_keys`（空白归一化后的字符串）做哈希，每组只留第一行。
- `MathVerifyFilter` 比较 `input_key` 与 `input_gold_key`。装了 `math_verify` 时先用它，然后是归一化后的精确文本，再然后是两边最后一个数（包括 `\boxed{...}`）。失败行默认丢掉，除非 `drop_failures=False`。布尔结果写在 `output_key`（默认 `math_correct`）。
- `SQLExecFilter` 是一个桩。受信任的 `setup_sql` 在每行各自的内存 sqlite 上执行。来自行内的 SQL 必须是单条 `SELECT` 或 `WITH ... SELECT`。写入语句和多语句算失败。结果最多保留 50 行。

## 📤 导出

`export_jsonl(path)` 写出最后一步，并附带 `*.manifest.json`（`kind: throughline-export`），里面有列、算子和血缘文本。

`export_argilla()` 在适配器落地之前抛出 `NotImplementedError`。

## 🎬 示例

`examples/pref_then_verify.py` 载入五条数学种子：一条重复的 `2 + 2`，一条写错的 `9 + 1` 草稿。`MockLLM` 把其余几条答对。评判器给全部五行打分，包括那条错的。哈希去重丢掉重复行。`MathVerifyFilter` 丢掉 `9 + 1`。剩下三行，写到 `examples/output/result.jsonl`。

这个顺序和文首的徽章一致：评判在客观门之前。以后的调度器可以把哈希和数学检查前移；v0 按 `forward` 记录的顺序执行。

## 🚫 v0 不做的事

v0 不跑 Ray，也不自带 OpenAI 或其他在线客户端。要接真实模型时，自己实现一个 `LLMClient`。测试和示例停在 `MockLLM`。

## 📚 参考文献

本仓库是接口层面的原型，不是下列工作的官方实现。

1. **UltraFeedback** — Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun。*UltraFeedback: Boosting Language Models with Scaled AI Feedback.* ICML 2024, PMLR 235。arXiv:2310.01377。  
   论文：[arxiv.org/abs/2310.01377](https://arxiv.org/abs/2310.01377) · 会议：[PMLR v235](https://proceedings.mlr.press/v235/cui24f.html) · 代码：[thunlp/UltraFeedback](https://github.com/thunlp/UltraFeedback)  
   评判方面（helpfulness、honesty、instruction-following、truthfulness，以及 overall-rating）来自这项工作。

相关训练目标，本仓库没有实现，但示例里的偏好分与核对列是为它们准备的：

2. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn。*Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023。arXiv:2305.18290。[论文](https://arxiv.org/abs/2305.18290)

3. Long Ouyang 等。*Training language models to follow instructions with human feedback.* NeurIPS 2022。arXiv:2203.02155。[论文](https://arxiv.org/abs/2203.02155)

4. Yizhong Wang 等。*Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023。arXiv:2212.10560。[论文](https://arxiv.org/abs/2212.10560)
