# A compact survey of modern NLP

> Working thesis: modern NLP is best understood not as a list of tasks, but as a sequence of increasingly general transfer regimes: static representations, contextual pretraining, generative foundation models, and retrieval-augmented systems. The field's main engineering question is no longer only "which architecture works?" but "where should knowledge live: in parameters, prompts, retrieved context, or tools?"

## Target reader

- Primary reader: ML engineers, research engineers, and graduate students entering NLP from general machine learning.
- Assumed background: linear algebra, optimization, standard supervised learning, and basic familiarity with neural networks.
- What the reader should be able to do after reading: choose a reasonable modeling regime for an NLP problem, identify the main evaluation risks, and understand where multilingual and retrieval considerations materially change system design.

## Scope

### In scope

- Text-centric NLP from word embeddings through transformer-era foundation models.
- The shift from task-specific models to transfer learning, instruction tuning, and retrieval augmentation.
- Cross-cutting concerns: multilingual transfer, benchmark design, behavioral evaluation, and deployment tradeoffs.

### Out of scope

- Speech-only pipelines, speech recognition, and text-to-speech.
- Vision-language and multimodal agents except where they clarify NLP trends.
- Exhaustive coverage of every subtask, every benchmark, or every commercial model family.

## Executive summary

Modern NLP has undergone three decisive compressions. First, word embeddings compressed lexical statistics into reusable vectors, making feature engineering less central. Second, pretrained contextual models compressed a wide range of linguistic regularities into transferable representations, which made downstream supervision cheaper and pushed model design toward large-scale pretraining plus lightweight adaptation. Third, generative language models and instruction tuning compressed many tasks into prompting interfaces, shifting the practical bottleneck from task-specific architecture to data curation, evaluation, and systems integration.

This evolution changed where practitioners should spend effort. For many classic classification, tagging, and retrieval problems, the high-value work is now task framing, data quality, and evaluation design rather than inventing new encoders. For knowledge-intensive tasks, pure parametric models are often insufficient: retrieval improves freshness, attribution, and controllability, but raises new ranking, latency, and provenance problems. For multilingual settings, scale helps, but language coverage and benchmark choice still dominate whether apparent progress transfers beyond English.

A useful taxonomy is therefore not chronological but architectural: **feature-centric systems**, **pretrained transfer systems**, and **augmented language systems**. Each regime moves more capability into pretraining, but also changes failure modes. As models become more general, held-out benchmark accuracy becomes less informative on its own, making behavioral testing, holistic evaluation, and explicit claim auditing central to credible NLP system design.

## Canonical papers and artifacts

| Item | Year | Why it matters | Section |
| --- | --- | --- | --- |
| [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) | 2013 | Established fast large-scale word embeddings as a reusable substrate for NLP. | Problem framing |
| [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural) | 2014 | Defined end-to-end sequence transduction as a general template for generation tasks. | Core approaches |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Replaced recurrence with attention and made scaling far easier. | Core approaches |
| [Deep Contextualized Word Representations](https://aclanthology.org/N18-1202/) | 2018 | Showed why context-dependent representations outperform static embeddings across tasks. | Core approaches |
| [BERT](https://aclanthology.org/N19-1423/) | 2019 | Made pretrain-then-finetune the dominant paradigm for NLU. | Core approaches |
| [T5](https://www.jmlr.org/papers/v21/20-074.html) | 2020 | Unified many NLP tasks as text-to-text transfer. | Core approaches |
| [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) | 2020 | Demonstrated large-scale in-context learning as a general interface. | Core approaches |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 2020 | Reintroduced explicit memory as a first-class component of language systems. | Systems and deployment considerations |
| [XTREME](https://proceedings.mlr.press/v119/hu20b.html) | 2020 | Formalized multilingual generalization as a cross-task benchmark problem. | Data and benchmarks |
| [HELM](https://arxiv.org/abs/2211.09110) | 2022 | Shifted evaluation from single-score leaderboards toward broader capability/risk reporting. | Evaluation |

## Taxonomy at a glance

The most useful organizing frame for NLP is **where generalization is stored**.

### Bucket 1: feature-centric systems

- definition: systems that rely on static embeddings, task-specific architectures, and task-specific supervision.
- representative papers: [word2vec](https://arxiv.org/abs/1301.3781), [seq2seq](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural)
- strengths: efficient, interpretable design boundaries, often easier to deploy in narrow settings.
- limits: weak transfer, heavy feature/task engineering, brittle across domains.

### Bucket 2: pretrained transfer systems

- definition: systems that pretrain large contextual encoders or encoder-decoder models and adapt them downstream.
- representative papers: [ELMo](https://aclanthology.org/N18-1202/), [BERT](https://aclanthology.org/N19-1423/), [T5](https://www.jmlr.org/papers/v21/20-074.html), [XLM-R](https://aclanthology.org/2020.acl-main.747/)
- strengths: strong sample efficiency, broad reuse across tasks, good default for supervised NLP.
- limits: knowledge can be stale, adaptation still depends on benchmark alignment, multilingual gains are uneven.

### Bucket 3: augmented language systems

- definition: systems that rely on large generative models plus prompting, instruction tuning, retrieval, or external tools.
- representative papers: [GPT-3](https://arxiv.org/pdf/2005.14165), [Self-Instruct](https://arxiv.org/abs/2212.10560), [Scaling Instruction-Finetuned Language Models](https://www.jmlr.org/papers/v25/23-0870.html), [RAG](https://arxiv.org/abs/2005.11401)
- strengths: flexible interfaces, low task-specific overhead, easier unification of generation and reasoning workflows.
- limits: attribution and factuality remain hard, evaluation is unstable, latency and cost can dominate production design.

## Main body

### 1. Problem framing

Early modern NLP systems were organized around the idea that lexical statistics could stand in for hand-built symbolic features. Word embeddings such as [word2vec](https://arxiv.org/abs/1301.3781) made this practical at web scale by learning reusable vector spaces from co-occurrence structure. This move mattered not only because embeddings improved word similarity scores, but because it changed the unit of reuse: once a lexical layer was pretrained, downstream models could inherit dense semantic information instead of relearning it from sparse indicators.

The next framing shift was from **word prediction** to **sequence transduction**. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural) made translation and related tasks look like general conditional generation problems. The later transformer transition then made the same idea scale better. From this point onward, the central question in NLP became how much task structure should be hard-coded versus absorbed by pretraining.

Today, NLP is less a collection of isolated tasks than a design space for text interfaces. The same model family may be used for classification, retrieval, summarization, question answering, extraction, or agentic tool use. That makes system design hinge on three questions: what prior knowledge is stored in parameters, what context is supplied at inference time, and how much task-specific supervision is still needed to make evaluation trustworthy.

### 2. Core approaches

The transformer era began with [Attention Is All You Need](https://arxiv.org/abs/1706.03762), which removed recurrent bottlenecks and made parallel pretraining practical. The immediate downstream effect was not only better translation quality, but a much cleaner scaling path for model size, context mixing, and transfer learning. [ELMo](https://aclanthology.org/N18-1202/) then demonstrated why contextual representations changed the field: one word type no longer required one fixed embedding, so downstream models could exploit context-sensitive lexical meaning.

[BERT](https://aclanthology.org/N19-1423/) turned this into the default supervised NLP recipe: pretrain a bidirectional encoder, fine-tune on downstream tasks, and reuse one backbone broadly. This regime worked especially well for classification, tagging, span extraction, and many leaderboard-style NLU tasks. [T5](https://www.jmlr.org/papers/v21/20-074.html) pushed the unification further by recasting diverse tasks in text-to-text form, making objective and interface design part of the same transfer framework.

The generative turn changed both capability and product surface. [GPT-3](https://arxiv.org/pdf/2005.14165) showed that enough scale can yield useful in-context adaptation without task-specific gradient updates. [Self-Instruct](https://arxiv.org/abs/2212.10560) and [Scaling Instruction-Finetuned Language Models](https://www.jmlr.org/papers/v25/23-0870.html) then showed that instruction tuning improves usability and generalization to unseen prompts and tasks. The tradeoff is that interface simplicity at inference time can hide training-data complexity: prompt-following behavior depends heavily on the instruction mixture, post-training data, and evaluation setup.

The important synthesis is that newer approaches do not strictly replace older ones. Encoder-style transfer models still dominate many narrow supervised pipelines because they are cheaper and easier to calibrate. Generative models dominate when task boundaries are fuzzy, outputs are open-ended, or one model must serve many interfaces. Retrieval-augmented systems dominate when freshness, provenance, or domain adaptation matter more than a purely parametric memory.

### 3. Data and benchmarks

Benchmark design has repeatedly shaped what the field believes it has solved. [GLUE](https://aclanthology.org/W18-5446/) helped crystallize language understanding as multi-task transfer. [SuperGLUE](https://proceedings.neurips.cc/paper/8589-superglue-a-stickier-benchmark-for-general-purpose-language-understanding-systems.pdf) raised the bar once GLUE saturated. But the broader lesson is not just that benchmarks get solved; it is that narrow aggregate scores can compress away important failure modes.

Multilingual NLP made this more obvious. [XTREME](https://proceedings.mlr.press/v119/hu20b.html) reframed progress as cross-lingual generalization across tasks and languages rather than English-only gains. [XLM-R](https://aclanthology.org/2020.acl-main.747/) showed that multilingual scaling can deliver strong transfer, including gains on lower-resource languages, but also exposed the tradeoff between positive transfer and capacity dilution. Strong average multilingual scores do not imply uniformly good performance for underrepresented languages, scripts, or domains.

For retrieval-heavy NLP, [BEIR](https://arxiv.org/abs/2104.08663) is a useful reminder that homogeneous benchmarks can flatter brittle methods. BEIR's zero-shot setting showed that classical lexical retrieval remains a robust baseline and that high-performing neural retrieval systems often pay meaningful compute costs for their gains. This is an important corrective to papers that report narrow-domain wins without testing distribution shift.

### 4. Evaluation

Held-out accuracy remains necessary, but it is no longer sufficient. [CheckList](https://arxiv.org/abs/2005.04118) made a software-testing analogy explicit: models that look good on benchmark averages can still fail simple behavioral tests. This is especially relevant for production NLP, where stakeholders care about consistency under paraphrase, negation, perturbation, and demographic variation, not only leaderboard rank.

The evaluation problem becomes sharper with large generative models. [HELM](https://arxiv.org/abs/2211.09110) argued for a broader reporting frame that includes accuracy, calibration, robustness, fairness-related concerns, efficiency, and transparency. This is the right direction because modern NLP systems do not fail in one way. A system can answer correctly yet be too slow, too expensive, insufficiently attributable, or too inconsistent under prompt variation.

A practical evaluation stack for NLP therefore needs at least three layers. First, benchmark evaluation for comparability. Second, behavioral or stress testing for known failure modes. Third, task-specific human review or offline replay for product realism. Any survey or system design that stops at benchmark averages is likely to overstate progress, especially in multilingual and knowledge-intensive settings.

### 5. Systems and deployment considerations

The defining systems question in current NLP is where to store and retrieve knowledge. Purely parametric models are elegant because inference is simple: the model maps input to output from internal weights alone. But [RAG](https://arxiv.org/abs/2005.11401) showed why this abstraction breaks for knowledge-intensive tasks. Retrieval allows models to incorporate fresher evidence, cite sources more naturally, and decouple some factual updates from full retraining.

This comes with real tradeoffs. Retrieval quality becomes part of model quality; indexing, chunking, and reranking become first-class design choices; and latency budgets must absorb search plus generation. In many production settings, a modest generator paired with strong retrieval and reranking is preferable to a larger parametric model. In others, retrieval is unnecessary overhead because the task is narrow, the ontology is fixed, or the target output is tightly constrained.

Instruction tuning and prompting also change deployment economics. They reduce the need for task-specific heads and pipelines, but can make behavior harder to pin down. Prompt formats become de facto APIs, and prompt drift can create silent regressions. This is one reason why encoder-style systems remain attractive for stable, audited workflows such as moderation, tagging, or triage, even when generative models appear more general.

### 6. Open problems

The first open problem is **evaluation realism**. Benchmarks saturate faster than deployment conditions stabilize, so the field still lacks a routine way to connect offline scores with operational reliability. The second is **multilingual equity**: strong English-centered or even average multilingual performance still masks substantial coverage gaps in low-resource languages, dialects, and specialized domains.

The third is **knowledge placement**. Retrieval, tool use, and longer-context models offer overlapping answers to the same problem: how to access information not safely stored in weights alone. The field still lacks crisp guidance on when to prefer larger context windows, stronger retrieval, more instruction tuning, or domain-specific continued pretraining.

The fourth is **controllability under open-ended generation**. Instruction tuning improves usability, but reliability under adversarial prompts, ambiguous goals, or distribution shift remains uneven. Progress here likely depends less on one new model class than on better data curation, evaluation design, and system decomposition.

## Comparison table

| Approach | Best at | Fails when | Data needs | Compute profile | Notes |
| --- | --- | --- | --- | --- | --- |
| Static embeddings + task-specific models | narrow supervised tasks with stable labels and limited budgets | domain shift, polysemy, long-range dependencies | moderate task-labeled data | low to moderate | still viable when the pipeline is fixed and interpretable |
| Pretrained encoders / text-to-text transfer | classification, tagging, extraction, summarization with clear supervision | stale knowledge, brittle transfer outside benchmark distributions | large pretraining corpora plus modest task labels | moderate for training, moderate at inference | best default when outputs are constrained |
| Generative instruction-tuned LMs | open-ended generation, task unification, rapid prototyping | prompt instability, attribution gaps, hallucination | massive pretraining plus instruction data | high | useful interface layer, not automatically the best subsystem everywhere |
| Retrieval-augmented generation | knowledge-intensive QA, grounded synthesis, enterprise search | poor retrieval, latency-constrained workflows, noisy corpora | document collection plus retrieval supervision or tuning | moderate to high, with retrieval overhead | moves part of generalization into indexes and retrievers |
| Multilingual pretrained models | cross-lingual transfer and shared infrastructure across languages | underrepresented scripts, domain mismatch, language imbalance | balanced multilingual corpora are highly valuable | moderate to high | average gains can hide poor tail-language performance |

## Claim audit

| Claim | Source(s) | Confidence | Keep? |
| --- | --- | --- | --- |
| Contextual pretraining replaced static embeddings as the default transfer substrate for supervised NLP. | [ELMo](https://aclanthology.org/N18-1202/), [BERT](https://aclanthology.org/N19-1423/), [T5](https://www.jmlr.org/papers/v21/20-074.html) | high | yes |
| Transformer scaling changed NLP more by improving transfer and interface unification than by solving one specific task. | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [T5](https://www.jmlr.org/papers/v21/20-074.html), [GPT-3](https://arxiv.org/pdf/2005.14165) | high | yes |
| Retrieval is best viewed as a systems choice about knowledge placement, not only as a model add-on. | [RAG](https://arxiv.org/abs/2005.11401), [BEIR](https://arxiv.org/abs/2104.08663) | medium-high | yes |
| English-centric benchmark progress overstates field-wide progress in multilingual NLP. | [XTREME](https://proceedings.mlr.press/v119/hu20b.html), [XLM-R](https://aclanthology.org/2020.acl-main.747/) | high | yes |
| Modern NLP evaluation needs benchmark, behavioral, and holistic layers. | [GLUE](https://aclanthology.org/W18-5446/), [CheckList](https://arxiv.org/abs/2005.04118), [HELM](https://arxiv.org/abs/2211.09110) | high | yes |

## Exclusions and deliberate omissions

This survey intentionally does not try to catalog every NLP subfield, every model family, or every domain benchmark. It omits speech, multimodal systems, and detailed treatment of areas such as information extraction, parsing, and dialogue as independent literatures. Those topics matter, but including them here would weaken the survey's organizing thesis, which is about transfer regimes and system design choices in text-centric NLP.

## Revision hypotheses queue

- Add a short section on how longer-context models compete with retrieval and when each wins.
- Add a domain adaptation subsection comparing continued pretraining, LoRA-style adaptation, and retrieval for specialized corpora.
- Add a short appendix mapping the survey taxonomy onto the resources already listed in the main repository README.
