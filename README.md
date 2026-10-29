<p align="center">
  <img alt="DataGovernance" src="https://img.shields.io/badge/DataGovernance-数据合成-111111?style=for-the-badge&logo=gitbook&logoColor=white" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" /></a>
  <a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-dataframe-150458?style=flat-square&logo=pandas&logoColor=white" /></a>
  <a href="https://docs.pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/pytest-offline-0A9EDC?style=flat-square&logo=pytest&logoColor=white" /></a>
  <img alt="no API key" src="https://img.shields.io/badge/API%20key-不需要-2EA44F?style=flat-square&logo=shieldsdotio&logoColor=white" />
</p>

<h1 align="center">🧵 DataGovernance</h1>

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

## 为什么要有这个项目

合成训练数据时，一张表要连续走过生成、校验和打分。NL2SQL 就是这样：先按库表写出 SQL，再执行看看能不能跑，然后把能跑的 SQL 写成自然语言问题，最后给问题和 SQL 打分。`sql`、`schema`、`question`、`overall` 都是在这些步骤里才出现的。

列名写错，或者某一步写出了一列、后面却没有人读，当时不会报错。要等整批生成、提问、打分都跑完，才发现分数打在空列上，或者执行器用错了表结构。模型调用已经花掉了。

DataGovernance 在第一次模型调用之前核对每一列谁写、谁读，哪一列写出来之后就断了。接不上就停，不开始花钱。

NL2SQL 示例用的就是这个核对。一份只有库名和建表语句的种子，变成带可执行 SQL、自然语言问题和 UltraFeedback 分数的训练行。SQL 合成的形状来自 DataFlow，提问和打分的形状来自 Distilabel。

## 从一份库表到打过分的行

种子在 `examples/data/nl2sql_seeds.jsonl`。每一行是一个库，只有 `db_id` 和 `schema`。图书馆、学校、商店各一行。图书馆的库表是：

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

流水线为每个库合成若干条 SQL，只留下能在这份 schema 上执行的 `SELECT`，再写成问题并打分。开始时没有 `sql`、`question`、`overall`。结束时，同一份图书馆库表变成多行，例如：

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

分数看的是问题和 SQL 是否对得上，不是 SQL 能不能跑。下面这条语句能执行，结果是 Dune，但评判认为它没有回答那个问题，总分是 1：

```json
{
  "db_id": "library",
  "sql": "SELECT id, title FROM books WHERE author = 'Herbert';",
  "sql_ok": true,
  "sql_result": [[1, "Dune"]],
  "question": "Which books have the author Herbert?",
  "overall": 1.0,
  "model_name": "gpt-5.4-mini"
}
```

`ratings` 和 `rationales` 按 helpfulness、honesty、instruction-following、truthfulness 展开。学校和商店是同样的结局：一份 schema 展开成多行带分数的问答。一次真实调用里，3 行种子变成 15 行。完整结果在 `examples/output/nl2sql/result.jsonl`。

示例读取本机 `DataGovernance/llm/config.toml`，用里面的模型生成和打分。把 `config.example.toml` 复制成 `config.toml` 再填 key；这个文件不入库。

```bash
python3 -m pip install -e ".[dev]"
python3 examples/nl2sql_synthesize.py
```

## 列先接上，再调用模型

算子挂在 `Pipeline` 上，`forward` 是一串 `run`。列参数以 `input_` 或 `output_` 开头。`compile()` 只记录这些列，不调用算子，也不调用模型。缺列会抛出 `KeyError`，并写明算子、参数和当时有哪些列。

```python
class NL2SQLSynthesize(Pipeline):
    def __init__(self, cache_path, llm):
        super().__init__()
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL("examples/data/nl2sql_seeds.jsonl")
        self.sql = SQLSynthesisOp(llm, prompt=load_prompt("nl2sql.synthesize"))
        self.exec = SQLExecFilter()
        self.question = TextGenerationOp(llm, task=SQLQuestionTask(load_prompt("nl2sql.question")))
        self.judge = UltraFeedbackOp(llm, prompt=load_prompt("nl2sql.judge"))
        self.dedup = HashDeduplicateFilter()

    def forward(self):
        self.load.run(self.storage.step())
        self.sql.run(self.storage.step(), input_schema_key="schema", input_db_id_key="db_id", output_sql_key="sql", generate_num=5)
        self.exec.run(self.storage.step(), input_key="sql", input_setup_key="schema")
        self.question.run(self.storage.step())
        self.judge.run(self.storage.step(), input_key="question", input_generation_key="sql")
        self.dedup.run(self.storage.step(), input_key="sql")
```

`pipeline.draw()` 在任何模型调用之前列出每一列谁写、谁读。`(none)` 是还没有后续读者的列：分数、执行结果、复杂度留在导出行上，后面不再读它们。

```text
pipeline: load:LoadJSONL -> sql:SQLSynthesisOp -> exec:SQLExecFilter -> question:TextGenerationOp -> judge:UltraFeedbackOp -> dedup:HashDeduplicateFilter
sql: sql:SQLSynthesisOp -> exec:SQLExecFilter, question:TextGenerationOp, judge:UltraFeedbackOp, dedup:HashDeduplicateFilter
question: question:TextGenerationOp -> judge:UltraFeedbackOp
schema: sql:SQLSynthesisOp -> exec:SQLExecFilter, question:TextGenerationOp
dead columns: db_id, sql_complexity, sql_ok, sql_result, sql_error, ratings, rationales, overall, model_name
```

`SQLQuestionTask` 用 Distilabel 的 `inputs`、`outputs`、`format_input`、`format_output` 声明自己的列。`UltraFeedbackOp` 读问题和 SQL，写出方面分和 `overall`。`SQLExecFilter` 与 `HashDeduplicateFilter` 不调用模型：前者只接受单条 `SELECT`，后者按 SQL 文本只留第一行。

数学和 NL2SQL 的提示词在 `DataGovernance/recipes/`，每个领域一个文件（`math.toml`、`nl2sql.toml`）。每一段的 `llm` 选择 `config.toml` 里的 `[llm.<名字>]`；不写就用文件顶部的默认模型。每个名字都写自己的 `base_url`、`timeout` 和 `temperature`，模型不会打到另一个模型的地址。`{列名}` 是这一步要读的列，`compile()` 在调用模型之前核对。数学例子的结果在 `examples/output/math_synthesize/`。

示例里的 `llm` 是读本机 `config.toml` 的 `ChatCompletionsClient`。测试仍可用 `MockLLM`。每一步写成 JSONL。`export_jsonl()` 导出最后一表，并附上血缘。每个示例的结果单独放在 `examples/output/<示例名>/` 下。

## 功能

这些是相对 DataFlow 和 Distilabel 已经接上的能力。NL2SQL 示例只用了其中生成、执行、提问和打分。下面这一节的例子把其余几项走一遍，不需要 API key。

- **先编译再运行。** `compile()` 记下每一列谁写、谁读，不调用模型。`draw()` 打印这张图。`run()` 才执行，并把已经完成且提示词、模型、温度都没变的步骤从缓存里拿回来。
- **每个模型自己的地址。** `config.toml` 里 `[llm.draft]` 和 `[llm.judge]` 各自写 `model`、`base_url`、`timeout`、`temperature`。`api_key` 和 `concurrency` 从文件顶部继承，某个模型要不同的值再在自己的表里写。`concurrency` 大于 1 时，多条对话同时发出，结果仍按请求顺序回来。
- **同一个提示词的多个版本。** `recipes/*.toml` 里的文本是版本 `current`。`register_prompt` 再登记一版，`use_prompt` 或 `load_prompt(name, version=...)` 选择用哪一版。流水线里的名字不用改。
- **一条指令多份回答，一次打分。** `TextGenerationOp(num_generations=2)` 把回答写成列表。`UltraFeedbackOp` 传入 `input_generations_key` 时，在同一次评判里给列表里的每一份打分。`ratings` 与回答一一对应。
- **按库名执行 SQL。** `DatabaseManager.register(db_id, path=文件)` 或 `register(db_id, schema=建表脚本)`。`SQLExecFilter(database=manager)` 配上 `input_db_id_key` 就连到那个库。种子里仍带着 `schema` 字符串时，不注册也能跑。
- **导出。** `export_jsonl()` 写结果和血缘清单。`export_argilla()` 写成标注用的 `instruction` 和 `generations`，不调用 Argilla 服务。`push_to_hub("owner/name")` 上传 `result.jsonl`，需要 `pip install 'DataGovernance[hub]'`。

## 一个走完这些功能的例子

`examples/feature_tour.py` 用 `MockLLM`。它登记两版提示词并选用 `strict`，对同一条算术题生成两份回答，在一次评判里打出 `[5.0, 2.0]`，再在本机 sqlite 文件上执行种子里的 `SELECT`，最后写出 JSONL、标注记录和一份本地的发布文件。

```bash
python3 examples/feature_tour.py
```

```python
register_prompt(
    "tour.generate",
    Prompt("{instruction}", system="Answer with one token."),
    version="plain",
)
register_prompt(
    "tour.generate",
    Prompt("Put only the number on the last line.\n{instruction}", system="Answer with one token."),
    version="strict",
)
use_prompt("tour.generate", "strict")

class FeatureTour(Pipeline):
    def __init__(self, cache_path, llm, database):
        super().__init__()
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL("examples/data/feature_tour.jsonl")
        self.gen = TextGenerationOp(llm, prompt=load_prompt("tour.generate"), num_generations=2)
        self.judge = UltraFeedbackOp(llm)
        self.exec = SQLExecFilter(database=database)
        self.dedup = HashDeduplicateFilter()

    def forward(self):
        self.load.run(self.storage.step())
        self.gen.run(self.storage.step())
        self.judge.run(
            self.storage.step(),
            input_generation_key=None,
            input_generations_key="generation",
            output_model="model_name",
        )
        self.exec.run(self.storage.step(), input_db_id_key="db_id", input_setup_key=None)
        self.dedup.run(self.storage.step(), input_key="sql")

pipe = FeatureTour(cache, llm, manager)
pipe.compile()
print(pipe.draw(), end="")
frame = pipe.run()
pipe.export_jsonl("examples/output/feature_tour/result.jsonl")
pipe.export_argilla("examples/output/feature_tour/review.jsonl")
```

跑完这一行是：`generation` 为 `["4", "four"]`，`ratings` 为 `[5.0, 2.0]`，`sql_result` 为 `[["Dune"]]`。文件在 `examples/output/feature_tour/`。

## 📚 参考文献

前面写的是要解决的问题和为此需要的行为。下面这些工作提供了接口形状或训练目标。本仓库不是它们的官方实现。

1. **DataFlow** — Hao Liang, Xiaochen Ma, Zhou Liu 等。*DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI.* arXiv:2512.16676, 2025。  
   论文：[arxiv.org/abs/2512.16676](https://arxiv.org/abs/2512.16676) · 代码：[OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow)  
   算子 `run(storage, input_*=..., output_*=...)`、按步写下的 JSONL，以及编译期列匹配，形状来自这篇工作。那边可以用 networkx 把列图画出来；这里的血缘停在文本，因为当前需要的是在调用模型之前看见断列。

2. **Distilabel** — Argilla 的数据合成框架（软件，不是论文）。  
   代码：[argilla-io/distilabel](https://github.com/argilla-io/distilabel) · 文档：[distilabel.argilla.io](https://distilabel.argilla.io/)  
   文本生成和偏好评判任务上的 `inputs`、`outputs`、`format_input`、`format_output` 与这份软件对齐。`distilabel.steps.tasks.UltraFeedback` 在一次提示里给一组 generation 打分。这里用 `input_generations_key` 做同一件事，`ratings` 是每份回答一个分数。`export_argilla()` 写出标注记录，`push_to_hub()` 对应 `Distiset.push_to_hub`，上传前要安装 `DataGovernance[hub]`。

3. **UltraFeedback** — Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun。*UltraFeedback: Boosting Language Models with Scaled AI Feedback.* ICML 2024, PMLR 235。arXiv:2310.01377。  
   论文：[arxiv.org/abs/2310.01377](https://arxiv.org/abs/2310.01377) · 会议：[PMLR v235](https://proceedings.mlr.press/v235/cui24f.html) · 代码：[thunlp/UltraFeedback](https://github.com/thunlp/UltraFeedback)  
   评判方面（helpfulness、honesty、instruction-following、truthfulness，以及 overall-rating）来自这项工作。

相关训练目标，本仓库没有实现，但示例里的偏好分与核对列是为它们准备的：

4. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn。*Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023。arXiv:2305.18290。[论文](https://arxiv.org/abs/2305.18290)

5. Long Ouyang 等。*Training language models to follow instructions with human feedback.* NeurIPS 2022。arXiv:2203.02155。[论文](https://arxiv.org/abs/2203.02155)

6. Yizhong Wang 等。*Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023。arXiv:2212.10560。[论文](https://arxiv.org/abs/2212.10560)
