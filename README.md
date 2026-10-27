<p align="center">
  <img alt="DataGovernance" src="https://img.shields.io/badge/DataGovernance-列的主线-111111?style=for-the-badge&logo=gitbook&logoColor=white" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" /></a>
  <a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-dataframe-150458?style=flat-square&logo=pandas&logoColor=white" /></a>
  <a href="https://docs.pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/pytest-offline-0A9EDC?style=flat-square&logo=pytest&logoColor=white" /></a>
  <img alt="no API key" src="https://img.shields.io/badge/API%20key-不需要-2EA44F?style=flat-square&logo=shieldsdotio&logoColor=white" />
</p>

<h1 align="center">🧵 DataGovernance</h1>

<p align="center">
  在任何一次模型调用之前，先把每一列的读写主线画出来。
</p>

<p align="center">
  <img alt="pipeline" src="https://img.shields.io/badge/①_Load-库表-4C6EF5?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="sql" src="https://img.shields.io/badge/②_SQL-合成-7950F2?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="exec" src="https://img.shields.io/badge/③_Exec-执行过滤-E03131?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="question" src="https://img.shields.io/badge/④_Question-提问-12B886?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="judge" src="https://img.shields.io/badge/⑤_Judge-打分-F59F00?style=flat-square" />
  <img alt="arrow" src="https://img.shields.io/badge/→-→-ffffff" />
  <img alt="dedupe" src="https://img.shields.io/badge/⑥_Dedupe-去重-0A9EDC?style=flat-square" />
</p>

## 问题

准备指令或偏好数据时，一张表要走过好几步。NL2SQL 这条流水线是：读入库表、按库表合成 SQL、丢掉不能执行的语句、为剩下的 SQL 写自然语言问题、给问题和 SQL 打分，再按 SQL 去掉重复。列是在这些步骤里一点点长出来的。

列名拼错，或者某一步写出了一列、后面却没有人读，往往要等整批生成跑完才发现。到那时，模型调用已经花掉了。

## 需求

在第一次模型调用之前，先看清每一列的读写主线：谁写它，谁读它，哪些列已经断了。

| | |
| --- | --- |
| 🧭 | 编译只记录 `input_*` / `output_*`，不调用 `operator.run()`，也不调用 LLM。列不存在就立刻失败 |
| 🧵 | 打印生产者 → 消费者，并标出没有任何后续读者的 **dead columns** |
| 🧪 | 哈希去重和只读 SQL 不经过模型。SQL 在该行自己的 schema 上执行 |
| 🔌 | 合成、提问和打分接在 `LLMClient` 上。`MockLLM` 可回放，不需要网络，也不需要 API key。真实模型读本机 `config.toml` |

**DataGovernance** 做的就是这件事。每一列是一条治理主线：谁写、谁读、哪里断了。`compile()` 在第一次模型调用之前把线画完。

每一步写成磁盘上的 JSONL。

## 📦 先跑起来

需要 Python 3.11+。在仓库根目录：

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 examples/nl2sql_synthesize.py --mock
```

`--mock` 不访问网络。它读 `examples/data/nl2sql_seeds.jsonl` 里的三份库表（图书馆、学校、商店），回放固定的 SQL、问题和分数。执行过滤丢掉不能执行的语句和 `INSERT`，去重丢掉重复 SQL。三个库各剩两行，一共六行。

要换成真实模型，把模板复制成本机配置再填 key。`config.toml` 已在 `.gitignore` 里，不会入库。

```bash
cp DataGovernance/llm/config.example.toml DataGovernance/llm/config.toml
python3 examples/nl2sql_synthesize.py
```

`pandas` 是表类型。`pytest` 跑测试，默认也不访问网络。`math-verify` 是可选额外依赖，这条 NL2SQL 路径不会 import 它。

```bash
python3 -m pip install -e ".[math]"   # 可选
```

## 一份库表会变成什么

`examples/data/nl2sql_seeds.jsonl` 里每一行是一个库，只有 `db_id` 和 `schema`。图书馆这一行的库表是：

```sql
CREATE TABLE books (
  id INTEGER PRIMARY KEY,
  title TEXT,
  year INTEGER,
  author TEXT
);
INSERT INTO books (id, title, year, author) VALUES
  (1, 'Dune', 1965, 'Herbert'),
  (2, 'Neuromancer', 1984, 'Gibson'),
  (3, 'Project Hail Mary', 2021, 'Weir');

CREATE TABLE loans (
  id INTEGER PRIMARY KEY,
  book_id INTEGER,
  borrower TEXT
);
INSERT INTO loans (id, book_id, borrower) VALUES
  (1, 1, 'Ada'),
  (2, 3, 'Grace');
```

学校、商店是另外两行，形状相同。流水线对每一个库各要 5 条 SQL，在这份 schema 上执行，丢掉执行失败的语句，再为留下的 SQL 写问题并打分。

用 `gpt-5.4-mini` 跑完后，图书馆这一行展开成 `examples/output/nl2sql/result.jsonl` 里的 5 行。下面两行是其中的头尾：一条满分，一条被判成不回答问题。`schema` 会原样抄到每一行上，这里省掉。

开始时没有 `sql`、`question`、`overall`。结束时一行是：

```json
{
  "db_id": "library",
  "sql": "SELECT id, title, year, author FROM books;",
  "sql_complexity": "easy",
  "sql_ok": true,
  "sql_result": [
    [1, "Dune", 1965, "Herbert"],
    [2, "Neuromancer", 1984, "Gibson"],
    [3, "Project Hail Mary", 2021, "Weir"]
  ],
  "question": "What are the id, title, year, and author of every book in the books table?",
  "overall": 5.0,
  "model_name": "gpt-5.4-mini"
}
```

同一份库表还会产出被打低分的行。评判读的是 `question` 和 `sql`：它认为下面这条 SQL 没有回答「Herbert 写了哪些书」。

```json
{
  "db_id": "library",
  "sql": "SELECT id, title FROM books WHERE author = 'Herbert';",
  "sql_ok": true,
  "sql_result": [[1, "Dune"]],
  "question": "Which books have the author Herbert?",
  "ratings": {
    "helpfulness": 1.0,
    "honesty": 1.0,
    "instruction_following": 1.0,
    "truthfulness": 1.0
  },
  "overall": 1.0,
  "model_name": "gpt-5.4-mini"
}
```

这次三个库都没有被执行过滤或去重删掉，所以 3 行种子变成 15 行结果。`--mock` 的结局是固定的：每个库先得到 5 条 SQL，执行丢掉缺列语句和 `INSERT`，去重再丢掉重复的那条，3 行种子变成 6 行。图书馆留下的两行是：

```json
{"db_id": "library", "sql_complexity": "easy", "question": "Which books were published in 1984 or later?", "sql": "SELECT title FROM books WHERE year >= 1984", "overall": 4.75, "model_name": "mock"}
{"db_id": "library", "sql_complexity": "medium", "question": "Which books are on loan, and who borrowed them?", "sql": "SELECT books.title, loans.borrower FROM books JOIN loans ON books.id = loans.book_id", "overall": 4.75, "model_name": "mock"}
```

## 🗺️ 目录

| 路径 | 作用 |
| --- | --- |
| `DataGovernance/core/operator.py` | `OperatorABC.run(storage, ...)` 与注册表 |
| `DataGovernance/core/storage.py` | `FileStorage` 的 JSONL 步骤链 |
| `DataGovernance/core/pipeline.py` | `compile` / `forward` / `run`、`AutoOP`、`KeyNode`、`draw` |
| `DataGovernance/llm/config.example.toml` | 配置模板，`api_key` 为空，可以入库 |
| `DataGovernance/llm/config.toml` | 本机配置。从模板复制后填写 key，已加入 `.gitignore` |
| `DataGovernance/llm/config.py` | `LLMConfig.load()` 读取 `config.toml` |
| `DataGovernance/llm/client.py` | `LLMClient` 协议、`MockLLM`、`ChatCompletionsClient` |
| `DataGovernance/ops/load.py` | `LoadJSONL` |
| `DataGovernance/ops/tasks.py` | 文本生成与偏好打分 |
| `DataGovernance/ops/nl2sql.py` | 由库表合成 SQL，再写成自然语言问题 |
| `DataGovernance/ops/verify.py` | 哈希去重、数学/精确核对、只读 SQL |
| `examples/data/nl2sql_seeds.jsonl` | 三份库表：`db_id` 和 `schema` |
| `examples/nl2sql_synthesize.py` | NL2SQL 合成、执行过滤、提问、打分、去重 |
| `examples/pref_then_verify.py` | 更短的数学例子 |
| `tests/` | `pytest`。默认不访问网络 |

## 🧩 这条流水线怎么写

继承 `Pipeline`，把唯一的 `FileStorage` 和各个算子放在 `self` 上，`forward` 写成一串 `run`。算子必须是流水线的属性，编译才能看见它们。NL2SQL 示例就是这个形状：

```python
class NL2SQLSynthesize(Pipeline):
    def __init__(self, cache_path, llm):
        super().__init__()
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL("examples/data/nl2sql_seeds.jsonl")
        self.sql = SQLSynthesisOp(llm)
        self.exec = SQLExecFilter()
        self.question = TextGenerationOp(llm, task=SQLQuestionTask())
        self.judge = UltraFeedbackOp(llm)
        self.dedup = HashDeduplicateFilter()

    def forward(self):
        self.load.run(self.storage.step())
        self.sql.run(
            self.storage.step(),
            input_schema_key="schema",
            input_db_id_key="db_id",
            output_sql_key="sql",
            output_complexity_key="sql_complexity",
            generate_num=5,
        )
        self.exec.run(
            self.storage.step(),
            input_key="sql",
            input_setup_key="schema",
        )
        self.question.run(self.storage.step())
        self.judge.run(
            self.storage.step(),
            input_key="question",
            input_generation_key="sql",
            output_model="model_name",
        )
        self.dedup.run(self.storage.step(), input_key="sql")
```

种子每行是一个库。`schema` 是建表和插入语句，`db_id` 是库名。

| 步骤 | 算子 | 读 | 写 |
| --- | --- | --- | --- |
| 合成 SQL | `SQLSynthesisOp` | `schema`、`db_id` | 每个库多行 `sql`，以及 `sql_complexity`（`easy` 或 `medium`） |
| 执行过滤 | `SQLExecFilter` | `sql`，并用 `schema` 当该行的 sqlite | `sql_ok`、`sql_result`、`sql_error`。不能执行的行丢掉 |
| 写问题 | `SQLQuestionTask` | `schema`、`sql` | `question`。这是 Distilabel 的 `inputs` / `outputs` / `format_input` / `format_output` |
| 打分 | `UltraFeedbackOp` | `question`、`sql` | `ratings`、`rationales`、`overall`、`model_name` |
| 去重 | `HashDeduplicateFilter` | `sql` | 同一条 SQL 只留第一行 |

`llm` 可以是 `MockLLM`，也可以是 `ChatCompletionsClient(LLMConfig.load())`。示例默认用后者。`--mock` 用前者。

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

NL2SQL 示例在调用模型之前打印出来是这样：

```text
pipeline: load:LoadJSONL -> sql:SQLSynthesisOp -> exec:SQLExecFilter -> question:TextGenerationOp -> judge:UltraFeedbackOp -> dedup:HashDeduplicateFilter
column lineage
db_id: sql:SQLSynthesisOp -> (none)
schema: sql:SQLSynthesisOp -> exec:SQLExecFilter, question:TextGenerationOp
sql: sql:SQLSynthesisOp -> exec:SQLExecFilter, question:TextGenerationOp, judge:UltraFeedbackOp, dedup:HashDeduplicateFilter
sql_complexity: sql:SQLSynthesisOp -> (none)
sql_ok: exec:SQLExecFilter -> (none)
sql_result: exec:SQLExecFilter -> (none)
sql_error: exec:SQLExecFilter -> (none)
question: question:TextGenerationOp -> judge:UltraFeedbackOp
ratings: judge:UltraFeedbackOp -> (none)
rationales: judge:UltraFeedbackOp -> (none)
overall: judge:UltraFeedbackOp -> (none)
model_name: judge:UltraFeedbackOp -> (none)
dead columns: db_id, sql_complexity, sql_ok, sql_result, sql_error, ratings, rationales, overall, model_name
```

`sql` 和 `schema`、`question` 后面还有人读。分数、执行结果和复杂度留在导出行上，后面的算子不再读它们，所以出现在 dead columns 里。血缘目前只打印这段文本。

## 🤖 生成、打分与模型客户端

需要生成或打分时，`LLMClient.generate(chats)` 接收一批对话（`{"role", "content"}` 消息），按顺序为每段对话返回一个字符串。`MockLLM` 回放一段脚本或一个 Python 函数，并记下每一次调用。没有网络。

地址和密钥写在 `DataGovernance/llm/config.toml`。仓库里提交的是 `config.example.toml`，`api_key` 为空。本机把模板复制成 `config.toml` 再填 key，这个文件不入库。`LLMConfig.load()` 读取本机文件。`api_key` 留空时配置仍能读出来，`generate` 调用 `require_api_key()` 后直接失败，不会发请求。

`ChatCompletionsClient` 只接收读出来的配置。每一段对话发一次 `POST` 到 `config.chat_completions_url`，按原顺序取 `choices[0].message.content`。

```toml
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = ""
timeout = 60
```

```python
from DataGovernance import ChatCompletionsClient, LLMConfig

llm = ChatCompletionsClient(LLMConfig.load())
```

`TextGenerationOp` 读 `input_key`（默认 `instruction`），写 `output_key`（默认 `generation`）。

`UltraFeedbackOp` 读一条 instruction 和一条 generation，按四个方面打 JSON 分：helpfulness、honesty、instruction-following、truthfulness，并写出：

- `ratings` — 方面 → 分数
- `rationales` — 方面 → 文字
- `overall` — 评判给出的总分，或各方面分数的平均

方面的来历，以及「一组 generation 各打一个分」的解析形状，写在参考文献里。算子本身每个样本只发送一条 generation。

也可以传入自带 `inputs`、`outputs`、`format_input`、`format_output` 的对象，由同一个客户端驱动。列名以对象自己的声明为准。

## ✅ 客观校验

这些算子不调用 LLM。

- `HashDeduplicateFilter` 对 `input_key` 或 `input_keys`（空白归一化后的字符串）做哈希，每组只留第一行。
- `MathVerifyFilter` 比较 `input_key` 与 `input_gold_key`。装了 `math_verify` 时先用它，然后是归一化后的精确文本，再然后是两边最后一个数（包括 `\boxed{...}`）。失败行默认丢掉，除非 `drop_failures=False`。布尔结果写在 `output_key`（默认 `math_correct`）。
- `SQLExecFilter` 在内存 sqlite 上执行。受信任的 `setup_sql` 对每一行相同；`input_setup_key` 则用该行自己的 schema。来自行内的 SQL 必须是单条 `SELECT` 或 `WITH ... SELECT`。写入语句和多语句算失败。结果最多保留 50 行。NL2SQL 示例用 `input_setup_key="schema"`。

## 📤 导出

`export_jsonl(path)` 写出最后一步，并附带 `*.manifest.json`（`kind: DataGovernance-export`），里面有列、算子和血缘文本。

`export_argilla()` 尚未实现，调用会抛出 `NotImplementedError`。标注平台要接的字段见参考文献。

## 🎬 跑完之后看哪里

真实模型跑完，最后一表在 `examples/output/nl2sql/result.jsonl`。同目录的 `result.manifest.json` 记下列、算子和血缘。六步中间表在 `examples/output/nl2sql/cache/step1.jsonl` 到 `step6.jsonl`。每次请求和响应在 `examples/output/nl2sql/live.log`，`Authorization` 已打码。

一次真实跑法（`gpt-5.4-mini`，每个库生成 5 条 SQL）的行数是 `3, 15, 15, 15, 15, 15`：三份库表进来，模型写出的 15 条 SQL 都能执行，去重没有再删行，最后 15 行。行数随模型输出变化。`--mock` 是固定的 `3, 15, 9, 9, 9, 6`，最后 6 行。

导出的一行长这样（schema 略去）：

| db_id | question | sql | overall | model_name |
| --- | --- | --- | --- | --- |
| library | Which books have the author Herbert? | `SELECT id, title FROM books WHERE author = 'Herbert';` | 1.0 | gpt-5.4-mini |
| library | What are the id, title, year, and author of every book in the books table? | `SELECT id, title, year, author FROM books;` | 5.0 | gpt-5.4-mini |

`overall` 是 UltraFeedback 的总分。同一条流水线里，问法和 SQL 对得上时可以是 5，评判认为 SQL 没有回答那个问题时可以是 1。`ratings` 和 `rationales` 按 helpfulness、honesty、instruction-following、truthfulness 展开。

`examples/pref_then_verify.py` 是更短的数学例子：五条种子、打分、去重，再按标准答案丢掉错行。它一直用 `MockLLM`。

## 🚫 不做的事

不跑分布式执行。`pipeline.run()` 仍在当前进程里按 `forward` 的顺序跑。在线模型用 `ChatCompletionsClient`。`pytest` 和 `--mock` 走 `MockLLM`。`examples/nl2sql_synthesize.py` 与 `tests/test_e2e.py` 在 `config.toml` 写了 `api_key` 时才请求真实模型。

## 📚 参考文献

前面写的是要解决的问题和为此需要的行为。下面这些工作提供了接口形状或训练目标。本仓库不是它们的官方实现。

1. **DataFlow** — Hao Liang, Xiaochen Ma, Zhou Liu 等。*DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI.* arXiv:2512.16676, 2025。  
   论文：[arxiv.org/abs/2512.16676](https://arxiv.org/abs/2512.16676) · 代码：[OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow)  
   算子 `run(storage, input_*=..., output_*=...)`、按步写下的 JSONL，以及编译期列匹配，形状来自这篇工作。那边可以用 networkx 把列图画出来；这里的血缘停在文本，因为当前需要的是在调用模型之前看见断列。

2. **Distilabel** — Argilla 的数据合成框架（软件，不是论文）。  
   代码：[argilla-io/distilabel](https://github.com/argilla-io/distilabel) · 文档：[distilabel.argilla.io](https://distilabel.argilla.io/)  
   文本生成和偏好评判任务上的 `inputs`、`outputs`、`format_input`、`format_output` 与这份软件对齐。`distilabel.steps.tasks.UltraFeedback` 在一次提示里给一组 generation 打分，返回平行的 `ratings` / `rationales` 列表和 `model_name`，方面可以是 `overall-rating`。这里的解析器接受这种浮点数列表，取平均写入 `overall`，给以后的适配器留位置。标注导出的目标是 `distilabel.steps.PreferenceToArgilla`；同一份工件上 Hugging Face Hub 的对应接口是 `Distiset.push_to_hub`。这两条接线都还没有做。

3. **UltraFeedback** — Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun。*UltraFeedback: Boosting Language Models with Scaled AI Feedback.* ICML 2024, PMLR 235。arXiv:2310.01377。  
   论文：[arxiv.org/abs/2310.01377](https://arxiv.org/abs/2310.01377) · 会议：[PMLR v235](https://proceedings.mlr.press/v235/cui24f.html) · 代码：[thunlp/UltraFeedback](https://github.com/thunlp/UltraFeedback)  
   评判方面（helpfulness、honesty、instruction-following、truthfulness，以及 overall-rating）来自这项工作。

相关训练目标，本仓库没有实现，但示例里的偏好分与核对列是为它们准备的：

4. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn。*Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023。arXiv:2305.18290。[论文](https://arxiv.org/abs/2305.18290)

5. Long Ouyang 等。*Training language models to follow instructions with human feedback.* NeurIPS 2022。arXiv:2203.02155。[论文](https://arxiv.org/abs/2203.02155)

6. Yizhong Wang 等。*Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023。arXiv:2212.10560。[论文](https://arxiv.org/abs/2212.10560)
