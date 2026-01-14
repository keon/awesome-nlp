# Awesome NLP

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

自然語言處理精選資源列表：從經典技術到大型語言模型。

![Awesome NLP Logo](/images/logo.jpg)

閱讀其他語言版本：[English](./README.md)、[繁體中文](./README-ZH-TW.md)

---

## 目錄

- [研究摘要與趨勢](#研究摘要與趨勢)
- [知名 NLP 研究實驗室](#知名-nlp-研究實驗室)
- [教學](#教學)
- [書籍](#書籍)
- [函式庫](#函式庫)
- [大型語言模型](#大型語言模型)
- [文本嵌入](#文本嵌入)
- [LLM 框架與工具](#llm-框架與工具)
- [智能代理](#智能代理)
- [RAG 檢索增強生成](#rag-檢索增強生成)
- [訓練與微調](#訓練與微調)
- [評估](#評估)
- [部署與服務](#部署與服務)
- [安全與防護](#安全與防護)
- [服務](#服務)
- [標註工具](#標註工具)
- [資料集](#資料集)
- [多語言 NLP](#多語言-nlp)
- [領域專用 NLP](#領域專用-nlp)
- [重要論文](#重要論文)

---

## 研究摘要與趨勢

* [NLP-Overview](https://nlpoverview.com/) - 深度學習技術應用於 NLP 的最新概述，包括理論、實作、應用和最先進的結果。
* [NLP-Progress](https://nlpprogress.com/) - 追蹤自然語言處理的進展，包括資料集和常見 NLP 任務的最新技術。
* [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
* [ACL 2018 Highlights](http://ruder.io/acl-2018-highlights/)
* [Four deep learning trends from ACL 2017 - Part One](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-1.html)
* [Four deep learning trends from ACL 2017 - Part Two](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-2.html)
* [Highlights of EMNLP 2017](http://blog.aylien.com/highlights-emnlp-2017-exciting-datasets-return-clusters/)
* [Deep Learning for NLP: Advancements & Trends](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/)
* [Survey of the State of the Art in Natural Language Generation](https://arxiv.org/abs/1703.09902)

---

## 知名 NLP 研究實驗室

### 學術界

* [Stanford NLP Group](https://nlp.stanford.edu/) - 頂尖 NLP 研究實驗室之一，[Stanford CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) 和 [Stanza](https://stanfordnlp.github.io/stanza/) 的創建者。
* [Berkeley NLP Group](http://nlp.cs.berkeley.edu/) - 以重建亞太地區 637 種語言的古老語言而聞名。
* [CMU Language Technologies Institute](https://www.lti.cs.cmu.edu/) - 知名專案包括瀕危語言的 [Avenue Project](http://www.cs.cmu.edu/~avenue/) 和 [Noah's Ark](http://www.cs.cmu.edu/~ark/)。
* [Johns Hopkins CLSP](http://clsp.jhu.edu/) - 語言與語音處理中心。
* [Columbia NLP Group](http://www1.cs.columbia.edu/nlp/index.cgi)
* [UMD CLIP](https://wiki.umiacs.umd.edu/clip/index.php/Main_Page) - 計算語言學和資訊處理。
* [Penn NLP](https://nlp.cis.upenn.edu/) - 以創建 [Penn Treebank](https://www.seas.upenn.edu/~pdtb/) 聞名。
* [Allen Institute for AI (AI2)](https://allenai.org/) - AllenNLP、Semantic Scholar、OLMo。
* [UW NLP](https://nlp.washington.edu/) - Noah Smith 的研究組。
* [ETH Zurich NLP](https://nlp.ethz.ch/) - Ryan Cotterell 的研究組。

### 產業界

* [OpenAI](https://openai.com/research) - GPT 系列、RLHF、推理模型。
* [Anthropic](https://www.anthropic.com/research) - Claude、Constitutional AI、可解釋性。
* [Google DeepMind](https://deepmind.google/research/) - Gemini、PaLM、AlphaCode。
* [Meta FAIR](https://ai.meta.com/research/) - Llama、NLLB、SeamlessM4T。
* [Mistral AI](https://mistral.ai/) - Mistral、Mixtral 模型。
* [Cohere](https://cohere.com/research) - 企業 NLP、Command R。

---

## 教學

### 閱讀內容 - 通用機器學習

* [Machine Learning 101](https://docs.google.com/presentation/d/1kSuQyW5DTnkVaZEjGYCkfOxvzCqGEFzWBy4e9Uedd9k/edit?usp=sharing) - Google 高級創意工程師為工程師和管理階層解釋機器學習。
* [AI Playbook](https://aiplaybook.a16z.com/) - a16z AI 劇本。
* [Ruder's Blog](http://ruder.io/#open) - Sebastian Ruder 對 NLP 研究的評論。
* [How To Label Data](https://www.lighttag.io/how-to-label-data/) - 管理語言標註專案的指南。
* [Depends on the Definition](https://www.depends-on-the-definition.com/) - 涵蓋 NLP 主題的部落格文章集合。

### 閱讀內容 - NLP 介紹與指南

* [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [NLP in Python](http://github.com/NirantK/nlp-python-deep-learning) - Github 筆記本集合。
* [Natural Language Processing: An Introduction](https://academic.oup.com/jamia/article/18/5/544/829676) - 牛津大學。
* [Deep Learning for NLP with Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
* [Hands-On NLTK Tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) - Jupyter 筆記本。
* [Natural Language Processing with Python](https://www.nltk.org/book/) - 使用 NLTK 介紹 NLP 的線上書籍。
* [Train a new language model from scratch](https://huggingface.co/blog/how-to-train) - Hugging Face 🤗
* [The Super Duper NLP Repo](https://notebooks.quantumstat.com/) - Colab 筆記本集合。

### 部落格與電子報

* [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) 和 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Natural Language Processing](https://nlpers.blogspot.com/) - Hal Daumé III。
* [arXiv: NLP (Almost) from Scratch](https://arxiv.org/pdf/1103.0398.pdf)
* [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness) - Karpathy。
* [Machine Learning Mastery: Deep Learning for NLP](https://machinelearningmastery.com/category/natural-language-processing)
* [Visual NLP Paper Summaries](https://amitness.com/categories/#nlp)
* [Ahead of AI](https://magazine.sebastianraschka.com/) - Sebastian Raschka。
* [Lil'Log](https://lilianweng.github.io/) - Lilian Weng。
* [The Gradient](https://thegradient.pub/)
* [Simon Willison's Weblog](https://simonwillison.net/)
* [Latent Space](https://www.latent.space/)
* [Chip Huyen's Blog](https://huyenchip.com/blog/)

### 影片與線上課程

* [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) - Richard Socher 和 Christopher Manning。
* [CMU CS 11-711: Advanced NLP](http://phontron.com/class/anlp2024/) - Graham Neubig。
* [UMass CS685: Advanced NLP](https://people.cs.umass.edu/~miyyer/cs685/)
* [Oxford Deep NLP](https://github.com/oxford-cs-deepnlp-2017/lectures)
* [CMU Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/)
* [Deep NLP Course by Yandex](https://github.com/yandexdataschool/nlp_course)
* [fast.ai NLP Course](https://www.fast.ai/2019/07/08/fastai-nlp/) - [筆記本](https://github.com/fastai/course-nlp)
* [AWS ML University - NLP](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw) - [教材](https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp)
* [Applied NLP - IIT Madras](https://www.youtube.com/playlist?list=PLH-xYrxjfO2WyR3pOAB006CYMhNt4wTqp) - [筆記本](https://github.com/Ramaseshanr/anlp)
* [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
* [DeepLearning.AI NLP Specialization](https://www.deeplearning.ai/courses/natural-language-processing-specialization/)

---

## 書籍

### 免費線上

* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - Jurafsky & Martin。
* [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class) - Jacob Eisenstein。
* [Text Mining in R](https://www.tidytextmining.com)
* [Natural Language Processing with Python](https://www.nltk.org/book/)

### 神經網路/LLM 時代

* [NLP with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - Hugging Face 團隊。
* [NLP with PyTorch](https://github.com/joosthub/PyTorchNLPBook)
* [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) - Sebastian Raschka。
* [Practical Natural Language Processing](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/)
* [Natural Language Processing with Spark NLP](https://www.oreilly.com/library/view/natural-language-processing/9781492047759/)
* [Deep Learning for Natural Language Processing](https://www.manning.com/books/deep-learning-for-natural-language-processing) - Stephan Raaijmakers。
* [Real-World Natural Language Processing](https://www.manning.com/books/real-world-natural-language-processing) - Masato Hagiwara。
* [Natural Language Processing in Action](https://www.manning.com/books/natural-language-processing-in-action-second-edition) - Hobson Lane。
* [Transformers in Action](https://www.manning.com/books/transformers-in-action) - Nicole Koenigstein。

---

## 函式庫

### Node.js 和 JavaScript

* [Twitter-text](https://github.com/twitter/twitter-text) - Twitter 的文本處理庫。
* [Knwl.js](https://github.com/benhmoore/Knwl.js) - JS 中的自然語言處理器。
* [Retext](https://github.com/retextjs/retext) - 可擴展的自然語言分析系統。
* [NLP Compromise](https://github.com/spencermountain/compromise) - 瀏覽器中的 NLP。
* [Natural](https://github.com/NaturalNode/natural) - Node 的通用 NLP 工具。
* [Poplar](https://github.com/synyi/poplar) - 基於網頁的標註工具。
* [NLP.js](https://github.com/axa-group/nlp.js) - 用於構建機器人的 NLP 庫。
* [node-question-answering](https://github.com/huggingface/node-question-answering) - Node.js 中使用 DistilBERT 的問答。

### Python

* [spaCy](https://github.com/explosion/spaCy) - 工業級 NLP :+1:
  * [textacy](https://github.com/chartbeat-labs/textacy) - 建立在 spaCy 上的高階 NLP。
* [NLTK](https://www.nltk.org/) - 自然語言工具包，50+ 語料庫。
* [Stanza](https://stanfordnlp.github.io/stanza/) - Stanford 的神經網路管線（70+ 語言）。
* [Flair](https://github.com/zalandoresearch/flair) - 最先進的 NLP，包含 BERT、ELMo、Flair 嵌入。
* [TextBlob](http://textblob.readthedocs.org/) - 常見 NLP 任務的簡單 API。
* [gensim](https://radimrehurek.com/gensim/index.html) - 無監督語義建模 :+1:
* [AllenNLP](https://github.com/allenai/allennlp) - 建立在 PyTorch 上的 NLP 研究庫。
* [Transformers](https://github.com/huggingface/transformers) - TensorFlow 2.0 和 PyTorch 的 NLP :+1:
* [Tokenizers](https://github.com/huggingface/tokenizers) - 用於研究和生產的快速分詞器。
* [Haystack](https://github.com/deepset-ai/haystack) - 端到端 NLP 框架。
* [PraisonAI](https://github.com/MervinPraison/PraisonAI) - 支援 100+ LLM 的多 AI 代理。
* [scattertext](https://github.com/JasonKessler/scattertext) - 語言差異的 d3 視覺化。
* [GluonNLP](https://github.com/dmlc/gluon-nlp) - MXNet 上的深度學習 NLP 工具包。
* [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP 研究工具包。
* [TextAttack](https://github.com/QData/TextAttack) - 對抗性攻擊和資料增強。
* [Kashgari](https://github.com/BrikerMan/Kashgari) - Keras 驅動的多語言 NLP。
* [FARM](https://github.com/deepset-ai/FARM) - 快速 NLP 遷移學習。
* [fairSeq](https://github.com/pytorch/fairseq) - Facebook AI seq2seq 模型。
* [Snips NLU](https://github.com/snipsco/snips-nlu) - 生產就緒的意圖解析。
* [NLP Architect](https://github.com/NervanaSystems/nlp-architect) - 最先進的 NLP 深度學習。
* [BigARTM](https://github.com/bigartm/bigartm) - 快速主題建模。
* [Sockeye](https://github.com/awslabs/sockeye) - 驅動 Amazon Translate 的神經機器翻譯。
* [DL Translate](https://github.com/xhlulu/dl-translate) - 50 種語言的翻譯。
* [Jury](https://github.com/obss/jury) - NLP 模型評估指標。
* [Rita DSL](https://github.com/zaibacu/rita-dsl) - 基於規則的 NLP 模式。
* [PyNLPl](https://github.com/proycon/pynlpl) - 通用 NLP 庫。
* [PySS3](https://github.com/sergioburdisso/pyss3) - 文本分類的白盒機器學習。
* [jPTDP](https://github.com/datquocnguyen/jPTDP) - 聯合詞性標註和依存解析（40+ 語言）。
* [Word Forms](https://github.com/gutfeeling/word_forms) - 生成英文單詞的所有形式。
* [Chazutsu](https://github.com/chakki-works/chazutsu) - 下載 NLP 研究資料集。
* [corex_topic](https://github.com/gregversteeg/corex_topic) - 階層式主題建模。

### C++

* [MIT Information Extraction Toolkit (MITIE)](https://github.com/mit-nlp/MITIE) - NER 和關係提取。
* [CRF++](https://taku910.github.io/crfpp/) - 條件隨機場實作。
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - 序列資料的 CRF。
* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - Charniak-Johnson 解析器。
* [colibri-core](https://github.com/proycon/colibri-core) - N-gram 和 skipgram。
* [ucto](https://github.com/LanguageMachines/ucto) - Unicode 感知分詞器。
* [frog](https://github.com/LanguageMachines/frog) - 荷蘭語的基於記憶的 NLP 套件。
* [MeTA](https://github.com/meta-toolkit/meta) - C++ 文本資料科學工具包。
* [Mecab](https://taku910.github.io/mecab/) - 日語形態分析器。
* [Moses](http://statmt.org/moses/) - 統計機器翻譯。
* [StarSpace](https://github.com/facebookresearch/StarSpace) - Facebook 嵌入庫。
* [InsNet](https://github.com/chncwang/InsNet) - 實例依賴的 NLP 模型。

### Java

* [Stanford NLP](https://nlp.stanford.edu/software/index.shtml)
* [OpenNLP](https://opennlp.apache.org/)
* [NLP4J](https://emorynlp.github.io/nlp4j/)
* [Word2vec in Java](https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec)
* [ReVerb](https://github.com/knowitall/reverb/) - 網路規模開放資訊提取。
* [OpenRegex](https://github.com/knowitall/openregex) - 基於 token 的正規表達式引擎。
* [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - 伊利諾大學 NLP 庫。
* [MALLET](http://mallet.cs.umass.edu/) - 文本機器學習：分類、聚類、主題建模。
* [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - 40+ 語言的詞性標註。

### Kotlin

* [Lingua](https://github.com/pemistahl/lingua/) - 長短文本的語言檢測。
* [Kotidgy](https://github.com/meiblorn/kotidgy) - 基於索引的文本資料生成器。

### Scala

* [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) - Apache Spark ML 上的 NLP。
* [Saul](https://github.com/CogComp/saul) - 包含 SRL、POS 模組的 NLP 系統。
* [ATR4S](https://github.com/ispras/atr4s) - 自動術語識別。
* [Epic](https://github.com/dlwh/epic) - 高性能統計解析器。
* [word2vec-scala](https://github.com/Refefer/word2vec-scala) - word2vec 的 Scala 介面。

### R

* [tidytext](https://github.com/juliasilge/tidytext) - 使用 tidy 工具的文本探勘。
* [text2vec](https://github.com/dselivanov/text2vec) - 向量化、主題建模、GloVe。
* [spacyr](https://github.com/quanteda/spacyr) - spaCy 的 R 包裝器。
* [wordVectors](https://github.com/bmschmidt/wordVectors) - word2vec 和嵌入。
* [RMallet](https://github.com/mimno/RMallet) - MALLET 的 R 介面。
* [corporaexplorer](https://kgjerde.github.io/corporaexplorer/) - 動態探索文本。
* [CRAN Task View: NLP](https://github.com/cran-task-views/NaturalLanguageProcessing/)

### Clojure

* [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp)
* [Infections-clj](https://github.com/r0man/inflections-clj) - Rails 風格的詞形變化。
* [postagga](https://github.com/fekr/postagga) - 解析自然語言。

### Ruby

* [ruby-nlp](https://github.com/diasks2/ruby-nlp) - NLP Ruby 庫集合。
* [nlp-with-ruby](https://github.com/arbox/nlp-with-ruby) - Ruby 實用 NLP。

### Rust

* [rust-bert](https://github.com/guillaume-be/rust-bert) - 基於 Transformer 的模型。
* [whatlang](https://github.com/greyblake/whatlang-rs) - 語言識別。
* [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) - 意圖解析。
* [adk-rust](https://github.com/zavora-ai/adk-rust) - AI 代理開發套件。

### NLP++

* [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=dehilster.nlp)
* [nlp-engine](https://github.com/VisualText/nlp-engine) - 包含英語解析器的 NLP++ 引擎。
* [VisualText](http://visualtext.org)

### Julia

* [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl)
* [TextModels.jl](https://github.com/JuliaText/TextModels.jl) - 神經網路模型。
* [WordTokenizers.jl](https://github.com/JuliaText/WordTokenizers.jl)
* [Word2Vec.jl](https://github.com/JuliaText/Word2Vec.jl)
* [Languages.jl](https://github.com/JuliaText/Languages.jl)
* [CorpusLoaders.jl](https://github.com/JuliaText/CorpusLoaders.jl)

---

## 大型語言模型

### 閉源模型

| 模型 | 提供商 | 上下文長度 |
|------|--------|------------|
| GPT-4 / GPT-4o / o1 / o3 | OpenAI | 128K |
| Claude 3.5 | Anthropic | 200K |
| Gemini 1.5/2.0 | Google | 1M+ |

### 開放權重模型

| 模型 | 提供商 | 參數量 |
|------|--------|--------|
| [Llama 3.1/3.2/3.3](https://llama.meta.com/) | Meta | 8B-405B |
| [Mistral/Mixtral](https://mistral.ai/) | Mistral AI | 7B-8x22B |
| [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) | 阿里巴巴 | 0.5B-72B |
| [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3) | DeepSeek | 671B MoE |
| [Yi](https://github.com/01-ai/Yi) | 01.AI | 6B-34B |
| [Falcon](https://huggingface.co/tiiuae) | TII | 7B-180B |
| [OLMo](https://allenai.org/olmo) | AI2 | 7B-65B |
| [Gemma 2](https://ai.google.dev/gemma) | Google | 2B-27B |

### 程式碼模型

- [Code Llama](https://github.com/meta-llama/codellama)
- [StarCoder 2](https://github.com/bigcode-project/starcoder2)
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)
- [Codestral](https://mistral.ai/news/codestral/)

### 架構變體

- [Mamba](https://github.com/state-spaces/mamba) - 狀態空間模型

### 神經網路 NLP 模型（LLM 之前）

**編碼器:** [BERT](https://github.com/google-research/bert) · [RoBERTa](https://arxiv.org/abs/1907.11692) · [DeBERTa](https://github.com/microsoft/DeBERTa) · [ALBERT](https://arxiv.org/abs/1909.11942) · [ELECTRA](https://arxiv.org/abs/2003.10555)

**多語言:** [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)（104 語言）· [XLM-R](https://arxiv.org/abs/1911.02116)（100 語言）

**編碼器-解碼器:** [T5](https://arxiv.org/abs/1910.10683) · [BART](https://arxiv.org/abs/1910.13461) · [mT5](https://arxiv.org/abs/2010.11934)

### 排行榜

- [Hugging Face Hub](https://huggingface.co/models)
- [LMSYS Chatbot Arena](https://arena.lmsys.org/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [Artificial Analysis](https://artificialanalysis.ai/)

---

## 文本嵌入

### 詞嵌入

**經驗法則:** fastText >> GloVe > word2vec

- [word2vec](https://arxiv.org/abs/1301.3781) · [實作](https://code.google.com/archive/p/word2vec/) · [解說](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [GloVe](https://nlp.stanford.edu/projects/glove/) · [論文](https://nlp.stanford.edu/pubs/glove.pdf) · [解說](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/)
- [fastText](https://fasttext.cc/) · [論文](https://arxiv.org/abs/1607.04606) · [解說](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)

### 句子與上下文嵌入

- [ELMo](https://arxiv.org/abs/1802.05365) · [PyTorch](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) · [TensorFlow](https://github.com/allenai/bilm-tf)
- [ULMFiT](https://arxiv.org/abs/1801.06146) - Jeremy Howard 和 Sebastian Ruder。
- [InferSent](https://arxiv.org/abs/1705.02364) - Facebook。
- [CoVe](https://arxiv.org/abs/1708.00107) - 上下文詞向量。
- [Paragraph Vectors](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) · [doc2vec 教學](https://rare-technologies.com/doc2vec-tutorial/)
- [sense2vec](https://arxiv.org/abs/1511.06388) - 詞義消歧。
- [Skip Thought Vectors](https://arxiv.org/abs/1506.06726)

### 現代嵌入模型

| 模型 | 提供商 |
|------|--------|
| [E5](https://huggingface.co/intfloat) | Microsoft |
| [BGE](https://huggingface.co/BAAI) | BAAI |
| [GTE](https://huggingface.co/Alibaba-NLP) | 阿里巴巴 |
| [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Nomic AI |
| [Jina Embeddings](https://huggingface.co/jinaai) | Jina AI |
| [Sentence Transformers](https://sbert.net/) | UKP Lab |
| [LaBSE](https://arxiv.org/abs/2007.01852) | Google（多語言）|
| [ColBERT](https://github.com/stanford-futuredata/ColBERT) | Stanford |
| [SPLADE](https://github.com/naver/splade) | Naver（稀疏）|

### 基準測試

- [MTEB](https://huggingface.co/spaces/mteb/leaderboard)
- [BEIR](https://github.com/beir-cellar/beir)

---

## LLM 框架與工具

### 應用框架

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [Haystack](https://github.com/deepset-ai/haystack)

### 結構化生成

- [Instructor](https://github.com/jxnl/instructor)
- [Outlines](https://github.com/outlines-dev/outlines)
- [Guidance](https://github.com/guidance-ai/guidance)
- [LMQL](https://github.com/eth-sri/lmql)

### Hugging Face 生態系統

- [transformers](https://github.com/huggingface/transformers)
- [tokenizers](https://github.com/huggingface/tokenizers)
- [datasets](https://github.com/huggingface/datasets)
- [accelerate](https://github.com/huggingface/accelerate)

### JavaScript/TypeScript

- [LangChain.js](https://github.com/langchain-ai/langchainjs)
- [Vercel AI SDK](https://github.com/vercel/ai)
- [Transformers.js](https://github.com/xenova/transformers.js)
- [LlamaIndex.TS](https://github.com/run-llama/LlamaIndexTS)

---

## 智能代理

### 框架

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [Smolagents](https://github.com/huggingface/smolagents)
- [PraisonAI](https://github.com/MervinPraison/PraisonAI)

### 程式碼代理

- [SWE-Agent](https://github.com/princeton-nlp/SWE-agent)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [Aider](https://github.com/paul-gauthier/aider)

### 基準測試

- [AgentBench](https://github.com/THUDM/AgentBench)
- [WebArena](https://webarena.dev/)
- [OSWorld](https://os-world.github.io/)

---

## RAG 檢索增強生成

### 框架

- [LlamaIndex](https://github.com/run-llama/llama_index)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Haystack](https://github.com/deepset-ai/haystack)
- [RAGFlow](https://github.com/infiniflow/ragflow)

### 向量資料庫

- [Pinecone](https://www.pinecone.io/)（託管）
- [Weaviate](https://weaviate.io/)
- [Qdrant](https://qdrant.tech/)
- [Chroma](https://www.trychroma.com/)
- [Milvus](https://milvus.io/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Elasticsearch](https://www.elastic.co/elasticsearch/vector-database)

### 重排序器

- [Cross-encoders](https://sbert.net/examples/applications/cross-encoder/README.html)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [Cohere Rerank](https://cohere.com/rerank)
- [Jina Reranker](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

### 評估

- [Ragas](https://github.com/explodinggradients/ragas)
- [ARES](https://github.com/stanford-futuredata/ARES)
- [TruLens](https://github.com/truera/trulens)

### 問答系統

- [DrQA](https://github.com/facebookresearch/DrQA) - Facebook Research 的維基百科問答。
- [Document-QA](https://github.com/allenai/document-qa) - AllenAI 的多段落閱讀理解。

---

## 訓練與微調

### PEFT 方法

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DoRA](https://arxiv.org/abs/2402.09353)

### 工具

- [PEFT](https://github.com/huggingface/peft)
- [trl](https://github.com/huggingface/trl)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

### 偏好優化

- DPO · KTO · IPO · ORPO

---

## 評估

### 框架

- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HELM](https://crfm.stanford.edu/helm/)
- [OpenAI Evals](https://github.com/openai/evals)
- [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai)
- [promptfoo](https://github.com/promptfoo/promptfoo)
- [DeepEval](https://github.com/confident-ai/deepeval)

### 基準測試

**通用:** MMLU · MMLU-Pro · ARC · HellaSwag

**推理:** GSM8K · MATH · BigBench-Hard · DROP

**程式碼:** HumanEval · MBPP · SWE-Bench · LiveCodeBench

**指令遵循:** MT-Bench · AlpacaEval · IFEval · Arena-Hard

**長上下文:** RULER · L-Eval · LongBench

**安全性:** TruthfulQA · HarmBench · JailbreakBench

---

## 部署與服務

### 推理框架

- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai/)
- [LM Studio](https://lmstudio.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang](https://github.com/sgl-project/sglang)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [ExecuTorch](https://github.com/pytorch/executorch)

### 託管推理

- [Together AI](https://together.ai/)
- [Fireworks AI](https://fireworks.ai/)
- [Replicate](https://replicate.com/)
- [Groq](https://groq.com/)
- [Modal](https://modal.com/)
- [Baseten](https://www.baseten.co/)

### 可觀測性

- [LangSmith](https://smith.langchain.com/)
- [LangFuse](https://langfuse.com/)
- [Arize Phoenix](https://phoenix.arize.com/)
- [Helicone](https://helicone.ai/)

---

## 安全與防護

- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- [LlamaGuard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/)
- [Lakera Guard](https://www.lakera.ai/)
- [Presidio](https://github.com/microsoft/presidio)（PII 檢測）
- [scrubadub](https://github.com/LeapBeyond/scrubadub)（PII 移除）

---

## 服務

NLP 作為具有更高級功能的 API：

- [OpenAI](https://platform.openai.com/) · [Anthropic](https://www.anthropic.com/api) · [Google](https://ai.google.dev/) · [Cohere](https://cohere.com/) · [Mistral](https://mistral.ai/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/) · [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) · [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Google Cloud NLP](https://cloud.google.com/natural-language/) · [AWS Comprehend](https://aws.amazon.com/comprehend/) · [Azure Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
- [Wit-ai](https://github.com/wit-ai/wit) - 自然語言介面。
- [IBM Watson NLU](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs)
- [ParallelDots](https://www.paralleldots.com/text-analysis-apis)
- [TextRazor](https://www.textrazor.com/)
- [Rosette](https://www.rosette.com/)
- [Textalytic](https://www.textalytic.com)
- [NLP Cloud](https://nlpcloud.io)
- [Cloudmersive](https://cloudmersive.com/nlp-api)
- [Vedika API](https://vedika.io)

---

## 標註工具

### 開源

- [Label Studio](https://labelstud.io/)
- [Argilla](https://github.com/argilla-io/argilla)
- [doccano](https://github.com/doccano/doccano)
- [brat](https://brat.nlplab.org/)
- [INCEpTION](https://inception-project.github.io/)
- [FLAT](https://github.com/proycon/flat)
- [Shoonya](https://github.com/AI4Bharat/Shoonya-Backend)
- [GATE](https://gate.ac.uk/overview.html)
- [Anafora](https://github.com/weitechen/anafora)
- [rstWeb](https://corpling.uis.georgetown.edu/rstweb/info/)
- [GitDox](https://corpling.uis.georgetown.edu/gitdox/)
- [Annotation Lab](https://www.johnsnowlabs.com/annotation-lab/)

### 商業

- [Prodigy](https://prodi.gy/) - 主動學習驅動。
- [LightTag](https://lighttag.io/)
- [Scale AI](https://scale.com/)
- [UBIAI](https://ubiai.tools/)
- [tagtog](https://www.tagtog.net/)
- [Datasaur](https://datasaur.ai/)
- [Konfuzio](https://konfuzio.com/en/)

---

## 資料集

### 資料庫

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)
- [nlp-datasets](https://github.com/niderhoff/nlp-datasets)
- [gensim-data](https://github.com/RaRe-Technologies/gensim-data)
- [tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp/)

### 預訓練

- [Common Crawl](https://commoncrawl.org/) · [The Pile](https://pile.eleuther.ai/) · [RedPajama](https://github.com/togethercomputer/RedPajama-Data) · [Dolma](https://github.com/allenai/dolma) · [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [The Stack](https://huggingface.co/datasets/bigcode/the-stack)（程式碼）

### 指令微調

- [FLAN Collection](https://github.com/google-research/FLAN) · [Natural Instructions](https://github.com/allenai/natural-instructions) · [P3](https://huggingface.co/datasets/bigscience/P3)
- [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) · [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) · [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) · [WizardLM](https://github.com/nlpxucan/WizardLM) · [Orca](https://arxiv.org/abs/2306.02707)

### 任務專用

- **問答:** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) · [Natural Questions](https://ai.google.com/research/NaturalQuestions) · [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) · [HotpotQA](https://hotpotqa.github.io/)
- **摘要:** [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) · [XSum](https://huggingface.co/datasets/xsum)
- **自然語言推理:** [SNLI](https://nlp.stanford.edu/projects/snli/) · [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) · [ANLI](https://github.com/facebookresearch/anli)
- **命名實體識別:** [CoNLL-2003](https://huggingface.co/datasets/conll2003) · [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19) · [WikiANN](https://huggingface.co/datasets/wikiann)
- **翻譯:** [WMT](https://www.statmt.org/wmt24/) · [OPUS](https://opus.nlpl.eu/) · [FLORES](https://github.com/facebookresearch/flores)

### 偏好

- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) · [SHP](https://huggingface.co/datasets/stanfordnlp/SHP) · [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)

---

## 多語言 NLP

### 多語言模型

- [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)（104 語言）
- [XLM-R](https://huggingface.co/xlm-roberta-base)（100 語言）
- [mT5](https://huggingface.co/google/mt5-base)（101 語言）
- [BLOOM](https://huggingface.co/bigscience/bloom)（46 語言）
- [Aya](https://huggingface.co/CohereForAI/aya-101)（101 語言）

### 翻譯

- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb)（200 語言）
- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication)

### 多語言框架

- [UDPipe](https://github.com/ufal/udpipe) - Universal Treebanks 的可訓練管線。
- [NLP-Cube](https://github.com/adobe/NLP-Cube) - 句子分割、分詞、詞性標註、解析。
- [UralicNLP](https://github.com/mikahama/uralicNLP) - 烏拉爾語和其他語言。
- [Stanza](https://stanfordnlp.github.io/stanza/)

---

<details>
<summary><strong>特定語言資源</strong></summary>

### 中文
**函式庫:** [jieba](https://github.com/fxsjy/jieba)、[SnowNLP](https://github.com/isnowfy/snownlp)、[HanLP](https://github.com/hankcs/HanLP)、[FudanNLP](https://github.com/FudanNLP/fnlp)
**模型:** [Qwen](https://github.com/QwenLM/Qwen)、[Yi](https://github.com/01-ai/Yi)、[ChatGLM](https://github.com/THUDM/ChatGLM-6B)、[Baichuan](https://github.com/baichuan-inc/Baichuan-7B)
**資源:** [funNLP](https://github.com/fighting41love/funNLP)

### 日文
**函式庫:** [MeCab](https://taku910.github.io/mecab/)、[SudachiPy](https://github.com/WorksApplications/SudachiPy)、[fugashi](https://github.com/polm/fugashi)
**資源:** [awesome-japanese-nlp](https://github.com/taishi-i/awesome-japanese-nlp-resources)

### 韓文
**函式庫:** [KoNLPy](http://konlpy.org)、[Mecab-ko](https://eunjeon.blogspot.com/)、[KoalaNLP](https://koalanlp.github.io/koalanlp/)、[KoNLP](https://cran.r-project.org/package=KoNLP)
**模型:** [KoBERT](https://github.com/SKTBrain/KoBERT)、[KoGPT](https://github.com/kakaobrain/kogpt)、[KULLM](https://github.com/nlpai-lab/KULLM)
**資料集:** [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus)、[NSMC](https://github.com/e9t/nsmc/)、[KorQuAD](https://korquad.github.io/)、[Korean Parallel Corpora](https://github.com/j-min/korean-parallel-corpora)
**教學:** [dsindex's blog](https://dsindex.github.io/)、[Kangwon NLP course](http://cs.kangwon.ac.kr/~leeck/NLP/)

### 阿拉伯語
**函式庫:** [goarabic](https://github.com/01walid/goarabic)、[jsastem](https://github.com/ejtaal/jsastem)、[PyArabic](https://pypi.org/project/PyArabic/)、[CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools)、[RFTokenizer](https://github.com/amir-zeldes/RFTokenizer)
**模型:** [AraBERT](https://github.com/aub-mind/arabert)、[Jais](https://huggingface.co/inception-mbzuai/jais-13b)
**資料集:** [LABR](https://github.com/mohamedadaly/labr)、[Arabic Stopwords](https://github.com/mohataher/arabic-stop-words)、[Multidomain Sentiment](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces)

### 印度語系
**函式庫:** [iNLTK](https://github.com/goru001/inltk)、[Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)、[Multi-Task DMA](https://github.com/Saurav0074/mt-dma)
**模型:** [IndicBERT](https://huggingface.co/ai4bharat/indic-bert)、[MuRIL](https://huggingface.co/google/muril-base-cased)、[Hindi2Vec](https://nirantk.com/hindi2vec/)、[Sanskrit Albert](https://huggingface.co/surajp/albert-base-sanskrit)
**資料集:** [Hindi Dependency Treebank](https://ltrc.iiit.ac.in/treebank_H2014/)、[BBC News Hindi](https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1)、[IIT Patna ABSA](https://github.com/pnisarg/ABSA)
**資源:** [AI4Bharat](https://ai4bharat.org/)

### 泰語
**函式庫:** [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)、[CutKum](https://github.com/pucktada/cutkum)、[SynThai](https://github.com/KenjiroAI/SynThai)、[JTCC](https://github.com/wittawatj/jtcc)
**模型:** [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
**資料:** [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm)

### 越南語
**函式庫:** [Underthesea](https://github.com/undertheseanlp/underthesea)、[VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP)、[vn.vitk](https://github.com/phuonglh/vn.vitk)、[pyvi](https://github.com/trungtv/pyvi)
**模型:** [PhoBERT](https://github.com/VinAIResearch/PhoBERT)
**資料集:** [Vietnamese treebank](https://vlsp.hpda.vn/demo/?page=resources&lang=en)、[BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf)、[VIVOS](https://ailab.hcmus.edu.vn/vivos/)、[ViText2SQL](https://github.com/VinAIResearch/ViText2SQL)、[EVB Corpus](https://github.com/qhungngo/EVBCorpus)

### 波斯語
**函式庫:** [Hazm](https://github.com/roshan-research/hazm)、[Parsivar](https://github.com/ICTRC/Parsivar)、[Perke](https://github.com/AlirezaTheH/perke)、[Perstem](https://github.com/jonsafari/perstem)、[virastar](https://github.com/aziz/virastar)
**模型:** [ParsBERT](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
**資料集:** [Bijankhan Corpus](https://dbrg.ut.ac.ir/بیژن%E2%80%8Cخان/)、[Uppsala Persian Corpus](https://sites.google.com/site/mojganserajicom/home/upc)、[LSCP](https://iasbs.ac.ir/~ansari/lscp/)、[ArmanPersoNERCorpus](https://github.com/HaniehP/PersianNER)、[PERLEX](http://farsbase.net/PERLEX.html)

### 印尼語
**函式庫:** [bahasa](https://github.com/kangfend/bahasa)、[Indonesian Word Embedding](https://github.com/galuhsahid/indonesian-word-embedding)
**模型:** [IndoBERT](https://github.com/indobenchmark/indonlu)
**資料集:** [IndoSum](https://github.com/kata-ai/indosum)、[Wordnet-Bahasa](http://wn-msa.sourceforge.net/)、[IndoNLU](https://github.com/indobenchmark/indonlu)

### 荷蘭語
**函式庫:** [python-frog](https://github.com/proycon/python-frog)、[Alpino](https://github.com/rug-compling/alpino)、[SimpleNLG_NL](https://github.com/rfdj/SimpleNLG-NL)、[Kaldi NL](https://github.com/opensource-spraakherkenning-nl/Kaldi_NL)
**模型:** [BERTje](https://github.com/wietsedv/bertje)、[RobBERT](https://github.com/iPieter/RobBERT)、[spaCy Dutch](https://spacy.io/models/nl)

### 西班牙語
**函式庫:** [spanlp](https://github.com/jfreddypuentes/spanlp)
**模型:** [BETO](https://github.com/dccuchile/beto)
**嵌入:** [Spanish Word Embeddings](https://github.com/dccuchile/spanish-word-embeddings)、[Spanish fastText](https://github.com/BotCenter/spanishWordEmbeddings)、[Spanish sent2vec](https://github.com/BotCenter/spanishSent2Vec)
**資料集:** [Columbian Political Speeches](https://github.com/dav009/LatinamericanTextResources)、[Copenhagen Treebank](https://mbkromann.github.io/copenhagen-dependency-treebank/)、[Spanish Billion Words](https://github.com/crscardellino/sbwce)

### 德語
- [German-NLP](https://github.com/adbar/German-NLP)

### 俄語
- [Natasha](https://github.com/natasha/natasha)、[pymorphy2](https://github.com/kmike/pymorphy2)、[DeepPavlov](https://github.com/deeppavlov/DeepPavlov)

### 波蘭語
- [Polish-NLP](https://github.com/ksopyla/awesome-nlp-polish)

### 葡萄牙語
- [Portuguese-NLP](https://github.com/ajdavidl/Portuguese-NLP)

### 烏克蘭語
- [awesome-ukrainian-nlp](https://github.com/asivokon/awesome-ukrainian-nlp)
- [UkrainianLT](https://github.com/Helsinki-NLP/UkrainianLT)

### 匈牙利語
- [awesome-hungarian-nlp](https://github.com/oroszgy/awesome-hungarian-nlp)

### 丹麥語
- [DaNLP](https://github.com/alexandrainst/danlp)、[daner](https://github.com/ITUnlp/daner)、[awesome-danish](https://github.com/fnielsen/awesome-danish)

### 烏爾都語
- [Urduhack](https://github.com/urduhack/urduhack)、[Urdu datasets](https://github.com/mirfan899/Urdu)

### 希伯來語
- [NLPH_Resources](https://github.com/NLPH/NLPH_Resources)

### 古典語言
- [CLTK](https://github.com/cltk/cltk) - 古典語言工具包。

### 亞洲語言（泰語、老撾語、中文、日文、韓文）
- ElasticSearch 中的 [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html)。

</details>

---

## 領域專用 NLP

### 生物醫學
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)、[BioBERT](https://github.com/dmis-lab/biobert)、[BioGPT](https://github.com/microsoft/BioGPT)、[ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [scispaCy](https://allenai.github.io/scispacy/)、[MedCAT](https://github.com/CogStack/MedCAT)

### 法律
- [LegalBERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)、[Saul-7B](https://huggingface.co/Equall/Saul-7B-Base)

### 金融
- [FinBERT](https://github.com/ProsusAI/finBERT)、[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)

### 科學
- [SciBERT](https://github.com/allenai/scibert)、[Galactica](https://huggingface.co/facebook/galactica-6.7b)
- [Semantic Scholar API](https://www.semanticscholar.org/product/api)

### 程式碼
- [CodeBERT](https://github.com/microsoft/CodeBERT)、[CodeT5](https://github.com/salesforce/CodeT5)、[StarCoder](https://github.com/bigcode-project/starcoder)

---

## 重要論文

**經典 NLP（1990s-2000s）**
- [A Maximum Entropy Approach to NLP](https://aclanthology.org/J96-1002/)（1996）
- [BLEU Score](https://aclanthology.org/P02-1040/)（2002）
- [Conditional Random Fields](https://repository.upenn.edu/cis_papers/159/)（2001）
- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/v3/blei03a.html)（2003）
- [A Unified Architecture for NLP](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf)（2008）

**神經網路 NLP（2013-2017）**
- [word2vec](https://arxiv.org/abs/1301.3781)（2013）
- [GloVe](https://aclanthology.org/D14-1162/)（2014）
- [Seq2Seq](https://arxiv.org/abs/1409.3215)（2014）
- [Attention](https://arxiv.org/abs/1409.0473)（2015）
- [ELMo](https://arxiv.org/abs/1802.05365)（2018）
- [ULMFiT](https://arxiv.org/abs/1801.06146)（2018）

**Transformer 時代（2017-2021）**
- [Transformer](https://arxiv.org/abs/1706.03762)（2017）
- [BERT](https://arxiv.org/abs/1810.04805)（2018）
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)（2019）
- [T5](https://arxiv.org/abs/1910.10683)（2019）
- [GPT-3](https://arxiv.org/abs/2005.14165)（2020）
- [Scaling Laws](https://arxiv.org/abs/2001.08361)（2020）
- [LoRA](https://arxiv.org/abs/2106.09685)（2021）

**LLM 時代（2022-2023）**
- [InstructGPT](https://arxiv.org/abs/2203.02155)（2022）
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)（2022）
- [LLaMA](https://arxiv.org/abs/2302.13971)（2023）
- [DPO](https://arxiv.org/abs/2305.18290)（2023）
- [QLoRA](https://arxiv.org/abs/2305.14314)（2023）

**2024**
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)（2024）
- [Mamba](https://arxiv.org/abs/2312.00752)（2024）
- [Llama 3](https://arxiv.org/abs/2407.21783)（2024）
- [Gemini 1.5](https://arxiv.org/abs/2403.05530)（2024）
- [Self-RAG](https://arxiv.org/abs/2310.11511)（2024）
- [Phi-3](https://arxiv.org/abs/2404.14219)（2024）

**2025-2026**
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)（2025）
- [Qwen2.5](https://arxiv.org/abs/2412.15115)（2025）
- [o1/o3 Reasoning](https://openai.com/index/learning-to-reason-with-llms/)（2025）
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)（2026）

---

## 相關列表

- [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning)
- [awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
- [awesome-llm](https://github.com/Hannibal046/Awesome-LLM)

---

## 貢獻

歡迎提交 PR 來新增資源、修復失效連結和更新內容。

---

## 授權

[CC0 1.0 Universal](./LICENSE)
