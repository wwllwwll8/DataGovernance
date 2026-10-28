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
        self.sql = SQLSynthesisOp(llm)
        self.exec = SQLExecFilter()
        self.question = TextGenerationOp(llm, task=SQLQuestionTask())
        self.judge = UltraFeedbackOp(llm)
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

`Pipeline` 上也可以直接 `load`、`generate`、`rank`、`score`，提示词里的 `{列名}` 就是这一步要读的列。NL2SQL 和 `examples/math_synthesize.py` 用的是同一套算子。数学例子的结果在 `examples/output/math_synthesize/`。

示例里的 `llm` 是读本机 `config.toml` 的 `ChatCompletionsClient`。测试仍可用 `MockLLM`。每一步写成 JSONL。`export_jsonl()` 导出最后一表，并附上血缘。每个示例的结果单独放在 `examples/output/<示例名>/` 下。

不跑分布式执行。标注平台导出还没有接上。

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
