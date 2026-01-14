# Awesome NLP

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of NLP resources: from classical techniques to large language models.

![Awesome NLP Logo](/images/logo.jpg)

Read this in [English](./README.md), [Traditional Chinese](./README-ZH-TW.md)

---

## Contents

- [Research Summaries and Trends](#research-summaries-and-trends)
- [Prominent NLP Research Labs](#prominent-nlp-research-labs)
- [Tutorials](#tutorials)
- [Books](#books)
- [Libraries](#libraries)
- [Large Language Models](#large-language-models)
- [Text Embeddings](#text-embeddings)
- [LLM Frameworks & Tools](#llm-frameworks--tools)
- [Agents](#agents)
- [RAG](#rag)
- [Training & Fine-tuning](#training--fine-tuning)
- [Evaluation](#evaluation)
- [Deployment & Serving](#deployment--serving)
- [Safety & Guardrails](#safety--guardrails)
- [Services](#services)
- [Annotation Tools](#annotation-tools)
- [Datasets](#datasets)
- [Multilingual NLP](#multilingual-nlp)
- [Domain-Specific NLP](#domain-specific-nlp)
- [Essential Papers](#essential-papers)

---

## Research Summaries and Trends

* [NLP-Overview](https://nlpoverview.com/) - Up-to-date overview of deep learning techniques applied to NLP, including theory, implementations, applications, and state-of-the-art results.
* [NLP-Progress](https://nlpprogress.com/) - Tracks the progress in Natural Language Processing, including datasets and current state-of-the-art for common NLP tasks.
* [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
* [ACL 2018 Highlights](http://ruder.io/acl-2018-highlights/)
* [Four deep learning trends from ACL 2017 - Part One](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-1.html)
* [Four deep learning trends from ACL 2017 - Part Two](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-2.html)
* [Highlights of EMNLP 2017](http://blog.aylien.com/highlights-emnlp-2017-exciting-datasets-return-clusters/)
* [Deep Learning for NLP: Advancements & Trends](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/)
* [Survey of the State of the Art in Natural Language Generation](https://arxiv.org/abs/1703.09902)

---

## Prominent NLP Research Labs

### Academic

* [Stanford NLP Group](https://nlp.stanford.edu/) - One of the top NLP research labs, creators of [Stanford CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) and [Stanza](https://stanfordnlp.github.io/stanza/).
* [Berkeley NLP Group](http://nlp.cs.berkeley.edu/) - Notable for reconstructing long dead languages from 637 languages in Asia and the Pacific.
* [CMU Language Technologies Institute](https://www.lti.cs.cmu.edu/) - Notable projects include [Avenue Project](http://www.cs.cmu.edu/~avenue/) for endangered languages and [Noah's Ark](http://www.cs.cmu.edu/~ark/).
* [Johns Hopkins CLSP](http://clsp.jhu.edu/) - Center for Language and Speech Processing.
* [Columbia NLP Group](http://www1.cs.columbia.edu/nlp/index.cgi)
* [UMD CLIP](https://wiki.umiacs.umd.edu/clip/index.php/Main_Page) - Computational Linguistics and Information Processing.
* [Penn NLP](https://nlp.cis.upenn.edu/) - Famous for creating the [Penn Treebank](https://www.seas.upenn.edu/~pdtb/).
* [Allen Institute for AI (AI2)](https://allenai.org/) - AllenNLP, Semantic Scholar, OLMo.
* [UW NLP](https://nlp.washington.edu/) - Noah Smith's group.
* [ETH Zurich NLP](https://nlp.ethz.ch/) - Ryan Cotterell's group.

### Industry

* [OpenAI](https://openai.com/research) - GPT series, RLHF, reasoning models.
* [Anthropic](https://www.anthropic.com/research) - Claude, Constitutional AI, interpretability.
* [Google DeepMind](https://deepmind.google/research/) - Gemini, PaLM, AlphaCode.
* [Meta FAIR](https://ai.meta.com/research/) - Llama, NLLB, SeamlessM4T.
* [Mistral AI](https://mistral.ai/) - Mistral, Mixtral models.
* [Cohere](https://cohere.com/research) - Enterprise NLP, Command R.

---

## Tutorials

### Reading Content - General Machine Learning

* [Machine Learning 101](https://docs.google.com/presentation/d/1kSuQyW5DTnkVaZEjGYCkfOxvzCqGEFzWBy4e9Uedd9k/edit?usp=sharing) - Google's Senior Creative Engineer explains ML for engineers and executives.
* [AI Playbook](https://aiplaybook.a16z.com/) - a16z AI playbook.
* [Ruder's Blog](http://ruder.io/#open) - Sebastian Ruder's commentary on NLP research.
* [How To Label Data](https://www.lighttag.io/how-to-label-data/) - Guide to managing linguistic annotation projects.
* [Depends on the Definition](https://www.depends-on-the-definition.com/) - Blog posts covering NLP topics with implementation.

### Reading Content - NLP Introductions and Guides

* [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [NLP in Python](http://github.com/NirantK/nlp-python-deep-learning) - Collection of Github notebooks.
* [Natural Language Processing: An Introduction](https://academic.oup.com/jamia/article/18/5/544/829676) - Oxford.
* [Deep Learning for NLP with Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
* [Hands-On NLTK Tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) - Jupyter notebooks.
* [Natural Language Processing with Python](https://www.nltk.org/book/) - Online book introducing NLP using NLTK.
* [Train a new language model from scratch](https://huggingface.co/blog/how-to-train) - Hugging Face 🤗
* [The Super Duper NLP Repo](https://notebooks.quantumstat.com/) - Collection of Colab notebooks.

### Blogs and Newsletters

* [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) and [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Natural Language Processing](https://nlpers.blogspot.com/) - Hal Daumé III.
* [arXiv: NLP (Almost) from Scratch](https://arxiv.org/pdf/1103.0398.pdf)
* [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness) - Karpathy.
* [Machine Learning Mastery: Deep Learning for NLP](https://machinelearningmastery.com/category/natural-language-processing)
* [Visual NLP Paper Summaries](https://amitness.com/categories/#nlp)
* [Ahead of AI](https://magazine.sebastianraschka.com/) - Sebastian Raschka.
* [Lil'Log](https://lilianweng.github.io/) - Lilian Weng.
* [The Gradient](https://thegradient.pub/)
* [Simon Willison's Weblog](https://simonwillison.net/)
* [Latent Space](https://www.latent.space/)
* [Chip Huyen's Blog](https://huyenchip.com/blog/)

### Videos and Online Courses

* [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) - Richard Socher and Christopher Manning.
* [CMU CS 11-711: Advanced NLP](http://phontron.com/class/anlp2024/) - Graham Neubig.
* [UMass CS685: Advanced NLP](https://people.cs.umass.edu/~miyyer/cs685/)
* [Oxford Deep NLP](https://github.com/oxford-cs-deepnlp-2017/lectures)
* [CMU Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/)
* [Deep NLP Course by Yandex](https://github.com/yandexdataschool/nlp_course)
* [fast.ai NLP Course](https://www.fast.ai/2019/07/08/fastai-nlp/) - [Notebooks](https://github.com/fastai/course-nlp)
* [AWS ML University - NLP](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw) - [Materials](https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp)
* [Applied NLP - IIT Madras](https://www.youtube.com/playlist?list=PLH-xYrxjfO2WyR3pOAB006CYMhNt4wTqp) - [Notebooks](https://github.com/Ramaseshanr/anlp)
* [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
* [DeepLearning.AI NLP Specialization](https://www.deeplearning.ai/courses/natural-language-processing-specialization/)

---

## Books

### Free Online

* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - Jurafsky & Martin.
* [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class) - Jacob Eisenstein.
* [Text Mining in R](https://www.tidytextmining.com)
* [Natural Language Processing with Python](https://www.nltk.org/book/)

### Neural/LLM Era

* [NLP with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - Hugging Face team.
* [NLP with PyTorch](https://github.com/joosthub/PyTorchNLPBook)
* [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) - Sebastian Raschka.
* [Practical Natural Language Processing](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/)
* [Natural Language Processing with Spark NLP](https://www.oreilly.com/library/view/natural-language-processing/9781492047759/)
* [Deep Learning for Natural Language Processing](https://www.manning.com/books/deep-learning-for-natural-language-processing) - Stephan Raaijmakers.
* [Real-World Natural Language Processing](https://www.manning.com/books/real-world-natural-language-processing) - Masato Hagiwara.
* [Natural Language Processing in Action](https://www.manning.com/books/natural-language-processing-in-action-second-edition) - Hobson Lane.
* [Transformers in Action](https://www.manning.com/books/transformers-in-action) - Nicole Koenigstein.

---

## Libraries

### Node.js and JavaScript

* [Twitter-text](https://github.com/twitter/twitter-text) - Twitter's text processing library.
* [Knwl.js](https://github.com/benhmoore/Knwl.js) - Natural Language Processor in JS.
* [Retext](https://github.com/retextjs/retext) - Extensible system for analyzing natural language.
* [NLP Compromise](https://github.com/spencermountain/compromise) - NLP in the browser.
* [Natural](https://github.com/NaturalNode/natural) - General NLP facilities for Node.
* [Poplar](https://github.com/synyi/poplar) - Web-based annotation tool.
* [NLP.js](https://github.com/axa-group/nlp.js) - NLP library for building bots.
* [node-question-answering](https://github.com/huggingface/node-question-answering) - QA with DistilBERT in Node.js.

### Python

* [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP :+1:
  * [textacy](https://github.com/chartbeat-labs/textacy) - Higher level NLP built on spaCy.
* [NLTK](https://www.nltk.org/) - Natural Language Toolkit, 50+ corpora.
* [Stanza](https://stanfordnlp.github.io/stanza/) - Stanford's neural pipeline (70+ languages).
* [Flair](https://github.com/zalandoresearch/flair) - State-of-the-art NLP with BERT, ELMo, Flair embeddings.
* [TextBlob](http://textblob.readthedocs.org/) - Simple API for common NLP tasks.
* [gensim](https://radimrehurek.com/gensim/index.html) - Unsupervised semantic modelling :+1:
* [AllenNLP](https://github.com/allenai/allennlp) - NLP research library built on PyTorch.
* [Transformers](https://github.com/huggingface/transformers) - NLP for TensorFlow 2.0 and PyTorch :+1:
* [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizers for research and production.
* [Haystack](https://github.com/deepset-ai/haystack) - End-to-end NLP framework with Transformers.
* [PraisonAI](https://github.com/MervinPraison/PraisonAI) - Multi-AI Agents with 100+ LLM support.
* [scattertext](https://github.com/JasonKessler/scattertext) - d3 visualizations of language differences.
* [GluonNLP](https://github.com/dmlc/gluon-nlp) - Deep learning toolkit for NLP on MXNet.
* [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP research toolkit.
* [TextAttack](https://github.com/QData/TextAttack) - Adversarial attacks and data augmentation.
* [Kashgari](https://github.com/BrikerMan/Kashgari) - Keras-powered multilingual NLP.
* [FARM](https://github.com/deepset-ai/FARM) - Fast transfer learning for NLP.
* [fairSeq](https://github.com/pytorch/fairseq) - Facebook AI seq2seq models.
* [Snips NLU](https://github.com/snipsco/snips-nlu) - Production ready intent parsing.
* [NLP Architect](https://github.com/NervanaSystems/nlp-architect) - State-of-the-art deep learning for NLP.
* [BigARTM](https://github.com/bigartm/bigartm) - Fast topic modelling.
* [Sockeye](https://github.com/awslabs/sockeye) - Neural MT powering Amazon Translate.
* [DL Translate](https://github.com/xhlulu/dl-translate) - Translation for 50 languages.
* [Jury](https://github.com/obss/jury) - NLP model evaluation metrics.
* [Rita DSL](https://github.com/zaibacu/rita-dsl) - Rule-based NLP patterns.
* [PyNLPl](https://github.com/proycon/pynlpl) - General purpose NLP library.
* [PySS3](https://github.com/sergioburdisso/pyss3) - White-box ML for text classification.
* [jPTDP](https://github.com/datquocnguyen/jPTDP) - Joint POS tagging and dependency parsing (40+ languages).
* [Word Forms](https://github.com/gutfeeling/word_forms) - Generate all forms of English words.
* [Chazutsu](https://github.com/chakki-works/chazutsu) - Download NLP research datasets.
* [sentimental-onix](https://github.com/sloev/sentimental-onix) - Sentiment models for spaCy.
* [Speedster](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster) - Optimization for inference speed.
* [corex_topic](https://github.com/gregversteeg/corex_topic) - Hierarchical topic modeling.

### C++

* [MIT Information Extraction Toolkit (MITIE)](https://github.com/mit-nlp/MITIE) - NER and relation extraction.
* [CRF++](https://taku910.github.io/crfpp/) - Conditional Random Fields implementation.
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFs for sequential data.
* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - Charniak-Johnson parser.
* [colibri-core](https://github.com/proycon/colibri-core) - N-grams and skipgrams.
* [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware tokenizer.
* [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite for Dutch.
* [MeTA](https://github.com/meta-toolkit/meta) - C++ Data Sciences Toolkit for text.
* [Mecab](https://taku910.github.io/mecab/) - Japanese morphological analyzer.
* [Moses](http://statmt.org/moses/) - Statistical machine translation.
* [StarSpace](https://github.com/facebookresearch/StarSpace) - Facebook embeddings library.
* [InsNet](https://github.com/chncwang/InsNet) - Instance-dependent NLP models.

### Java

* [Stanford NLP](https://nlp.stanford.edu/software/index.shtml)
* [OpenNLP](https://opennlp.apache.org/)
* [NLP4J](https://emorynlp.github.io/nlp4j/)
* [Word2vec in Java](https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec)
* [ReVerb](https://github.com/knowitall/reverb/) - Web-Scale Open Information Extraction.
* [OpenRegex](https://github.com/knowitall/openregex) - Token-based regex engine.
* [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - U of Illinois NLP libraries.
* [MALLET](http://mallet.cs.umass.edu/) - ML for text: classification, clustering, topic modeling.
* [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - POS tagging for 40+ languages.

### Kotlin

* [Lingua](https://github.com/pemistahl/lingua/) - Language detection for long and short text.
* [Kotidgy](https://github.com/meiblorn/kotidgy) - Index-based text data generator.

### Scala

* [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) - NLP on Apache Spark ML.
* [Saul](https://github.com/CogComp/saul) - NLP systems with SRL, POS modules.
* [ATR4S](https://github.com/ispras/atr4s) - Automatic term recognition.
* [Epic](https://github.com/dlwh/epic) - High performance statistical parser.
* [word2vec-scala](https://github.com/Refefer/word2vec-scala) - Scala interface to word2vec.

### R

* [tidytext](https://github.com/juliasilge/tidytext) - Text mining using tidy tools.
* [text2vec](https://github.com/dselivanov/text2vec) - Vectorization, topic modeling, GloVe.
* [spacyr](https://github.com/quanteda/spacyr) - R wrapper to spaCy.
* [wordVectors](https://github.com/bmschmidt/wordVectors) - word2vec and embeddings.
* [RMallet](https://github.com/mimno/RMallet) - R interface to MALLET.
* [corporaexplorer](https://kgjerde.github.io/corporaexplorer/) - Dynamic exploration of text.
* [CRAN Task View: NLP](https://github.com/cran-task-views/NaturalLanguageProcessing/)

### Clojure

* [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp)
* [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflections.
* [postagga](https://github.com/fekr/postagga) - Parse natural language.

### Ruby

* [ruby-nlp](https://github.com/diasks2/ruby-nlp) - Collection of NLP Ruby libraries.
* [nlp-with-ruby](https://github.com/arbox/nlp-with-ruby) - Practical NLP in Ruby.

### Rust

* [rust-bert](https://github.com/guillaume-be/rust-bert) - Transformer-based models.
* [whatlang](https://github.com/greyblake/whatlang-rs) - Language recognition.
* [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) - Intent parsing.
* [adk-rust](https://github.com/zavora-ai/adk-rust) - AI agent development kit.

### NLP++

* [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=dehilster.nlp)
* [nlp-engine](https://github.com/VisualText/nlp-engine) - NLP++ engine with English parser.
* [VisualText](http://visualtext.org)

### Julia

* [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl)
* [TextModels.jl](https://github.com/JuliaText/TextModels.jl) - Neural models.
* [WordTokenizers.jl](https://github.com/JuliaText/WordTokenizers.jl)
* [Word2Vec.jl](https://github.com/JuliaText/Word2Vec.jl)
* [Languages.jl](https://github.com/JuliaText/Languages.jl)
* [CorpusLoaders.jl](https://github.com/JuliaText/CorpusLoaders.jl)

---

## Large Language Models

### Closed-Source

| Model | Provider | Context |
|-------|----------|---------|
| GPT-4 / GPT-4o / o1 / o3 | OpenAI | 128K |
| Claude 3.5 | Anthropic | 200K |
| Gemini 1.5/2.0 | Google | 1M+ |

### Open-Weight

| Model | Provider | Parameters |
|-------|----------|------------|
| [Llama 3.1/3.2/3.3](https://llama.meta.com/) | Meta | 8B-405B |
| [Mistral/Mixtral](https://mistral.ai/) | Mistral AI | 7B-8x22B |
| [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) | Alibaba | 0.5B-72B |
| [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3) | DeepSeek | 671B MoE |
| [Yi](https://github.com/01-ai/Yi) | 01.AI | 6B-34B |
| [Falcon](https://huggingface.co/tiiuae) | TII | 7B-180B |
| [OLMo](https://allenai.org/olmo) | AI2 | 7B-65B |
| [Gemma 2](https://ai.google.dev/gemma) | Google | 2B-27B |

### Code Models

- [Code Llama](https://github.com/meta-llama/codellama)
- [StarCoder 2](https://github.com/bigcode-project/starcoder2)
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)
- [Codestral](https://mistral.ai/news/codestral/)

### Architecture Variants

- [Mamba](https://github.com/state-spaces/mamba) - State-space model

### Neural NLP Models (Pre-LLM)

**Encoders:** [BERT](https://github.com/google-research/bert) · [RoBERTa](https://arxiv.org/abs/1907.11692) · [DeBERTa](https://github.com/microsoft/DeBERTa) · [ALBERT](https://arxiv.org/abs/1909.11942) · [ELECTRA](https://arxiv.org/abs/2003.10555)

**Multilingual:** [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) (104 languages) · [XLM-R](https://arxiv.org/abs/1911.02116) (100 languages)

**Encoder-Decoder:** [T5](https://arxiv.org/abs/1910.10683) · [BART](https://arxiv.org/abs/1910.13461) · [mT5](https://arxiv.org/abs/2010.11934)

### Leaderboards

- [Hugging Face Hub](https://huggingface.co/models)
- [LMSYS Chatbot Arena](https://arena.lmsys.org/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [Artificial Analysis](https://artificialanalysis.ai/)

---

## Text Embeddings

### Word Embeddings

**Thumb Rule:** fastText >> GloVe > word2vec

- [word2vec](https://arxiv.org/abs/1301.3781) · [implementation](https://code.google.com/archive/p/word2vec/) · [explainer](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [GloVe](https://nlp.stanford.edu/projects/glove/) · [paper](https://nlp.stanford.edu/pubs/glove.pdf) · [explainer](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/)
- [fastText](https://fasttext.cc/) · [paper](https://arxiv.org/abs/1607.04606) · [explainer](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)

### Sentence & Contextual Embeddings

- [ELMo](https://arxiv.org/abs/1802.05365) · [PyTorch](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) · [TensorFlow](https://github.com/allenai/bilm-tf)
- [ULMFiT](https://arxiv.org/abs/1801.06146) - Jeremy Howard and Sebastian Ruder.
- [InferSent](https://arxiv.org/abs/1705.02364) - Facebook.
- [CoVe](https://arxiv.org/abs/1708.00107) - Contextualized Word Vectors.
- [Paragraph Vectors](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) · [doc2vec tutorial](https://rare-technologies.com/doc2vec-tutorial/)
- [sense2vec](https://arxiv.org/abs/1511.06388) - Word sense disambiguation.
- [Skip Thought Vectors](https://arxiv.org/abs/1506.06726)

### Modern Embedding Models

| Model | Provider |
|-------|----------|
| [E5](https://huggingface.co/intfloat) | Microsoft |
| [BGE](https://huggingface.co/BAAI) | BAAI |
| [GTE](https://huggingface.co/Alibaba-NLP) | Alibaba |
| [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Nomic AI |
| [Jina Embeddings](https://huggingface.co/jinaai) | Jina AI |
| [Sentence Transformers](https://sbert.net/) | UKP Lab |
| [LaBSE](https://arxiv.org/abs/2007.01852) | Google (multilingual) |
| [ColBERT](https://github.com/stanford-futuredata/ColBERT) | Stanford |
| [SPLADE](https://github.com/naver/splade) | Naver (sparse) |

### Benchmarks

- [MTEB](https://huggingface.co/spaces/mteb/leaderboard)
- [BEIR](https://github.com/beir-cellar/beir)

---

## LLM Frameworks & Tools

### Application Frameworks

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [Haystack](https://github.com/deepset-ai/haystack)

### Structured Generation

- [Instructor](https://github.com/jxnl/instructor)
- [Outlines](https://github.com/outlines-dev/outlines)
- [Guidance](https://github.com/guidance-ai/guidance)
- [LMQL](https://github.com/eth-sri/lmql)

### Hugging Face Ecosystem

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

## Agents

### Frameworks

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [Smolagents](https://github.com/huggingface/smolagents)
- [PraisonAI](https://github.com/MervinPraison/PraisonAI)

### Code Agents

- [SWE-Agent](https://github.com/princeton-nlp/SWE-agent)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [Aider](https://github.com/paul-gauthier/aider)

### Benchmarks

- [AgentBench](https://github.com/THUDM/AgentBench)
- [WebArena](https://webarena.dev/)
- [OSWorld](https://os-world.github.io/)

---

## RAG

### Frameworks

- [LlamaIndex](https://github.com/run-llama/llama_index)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Haystack](https://github.com/deepset-ai/haystack)
- [RAGFlow](https://github.com/infiniflow/ragflow)

### Vector Databases

- [Pinecone](https://www.pinecone.io/) (managed)
- [Weaviate](https://weaviate.io/)
- [Qdrant](https://qdrant.tech/)
- [Chroma](https://www.trychroma.com/)
- [Milvus](https://milvus.io/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Elasticsearch](https://www.elastic.co/elasticsearch/vector-database)

### Rerankers

- [Cross-encoders](https://sbert.net/examples/applications/cross-encoder/README.html)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [Cohere Rerank](https://cohere.com/rerank)
- [Jina Reranker](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

### Evaluation

- [Ragas](https://github.com/explodinggradients/ragas)
- [ARES](https://github.com/stanford-futuredata/ARES)
- [TruLens](https://github.com/truera/trulens)

### Question Answering Systems

- [DrQA](https://github.com/facebookresearch/DrQA) - Facebook Research on Wikipedia.
- [Document-QA](https://github.com/allenai/document-qa) - Multi-Paragraph Reading Comprehension by AllenAI.

---

## Training & Fine-tuning

### PEFT Methods

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DoRA](https://arxiv.org/abs/2402.09353)

### Tools

- [PEFT](https://github.com/huggingface/peft)
- [trl](https://github.com/huggingface/trl)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

### Preference Optimization

- DPO · KTO · IPO · ORPO

---

## Evaluation

### Frameworks

- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HELM](https://crfm.stanford.edu/helm/)
- [OpenAI Evals](https://github.com/openai/evals)
- [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai)
- [promptfoo](https://github.com/promptfoo/promptfoo)
- [DeepEval](https://github.com/confident-ai/deepeval)

### Benchmarks

**General:** MMLU · MMLU-Pro · ARC · HellaSwag

**Reasoning:** GSM8K · MATH · BigBench-Hard · DROP

**Code:** HumanEval · MBPP · SWE-Bench · LiveCodeBench

**Instruction:** MT-Bench · AlpacaEval · IFEval · Arena-Hard

**Long Context:** RULER · L-Eval · LongBench

**Safety:** TruthfulQA · HarmBench · JailbreakBench

---

## Deployment & Serving

### Inference Frameworks

- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai/)
- [LM Studio](https://lmstudio.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang](https://github.com/sgl-project/sglang)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [ExecuTorch](https://github.com/pytorch/executorch)

### Managed Inference

- [Together AI](https://together.ai/)
- [Fireworks AI](https://fireworks.ai/)
- [Replicate](https://replicate.com/)
- [Groq](https://groq.com/)
- [Modal](https://modal.com/)
- [Baseten](https://www.baseten.co/)

### Observability

- [LangSmith](https://smith.langchain.com/)
- [LangFuse](https://langfuse.com/)
- [Arize Phoenix](https://phoenix.arize.com/)
- [Helicone](https://helicone.ai/)

---

## Safety & Guardrails

- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- [LlamaGuard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/)
- [Lakera Guard](https://www.lakera.ai/)
- [Presidio](https://github.com/microsoft/presidio) (PII detection)
- [scrubadub](https://github.com/LeapBeyond/scrubadub) (PII removal)

---

## Services

NLP as API with higher level functionality:

- [OpenAI](https://platform.openai.com/) · [Anthropic](https://www.anthropic.com/api) · [Google](https://ai.google.dev/) · [Cohere](https://cohere.com/) · [Mistral](https://mistral.ai/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/) · [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) · [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Google Cloud NLP](https://cloud.google.com/natural-language/) · [AWS Comprehend](https://aws.amazon.com/comprehend/) · [Azure Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
- [Wit-ai](https://github.com/wit-ai/wit) - Natural Language Interface.
- [IBM Watson NLU](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs)
- [ParallelDots](https://www.paralleldots.com/text-analysis-apis)
- [TextRazor](https://www.textrazor.com/)
- [Rosette](https://www.rosette.com/)
- [Textalytic](https://www.textalytic.com)
- [NLP Cloud](https://nlpcloud.io)
- [Cloudmersive](https://cloudmersive.com/nlp-api)
- [Vedika API](https://vedika.io)

---

## Annotation Tools

### Open Source

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

### Commercial

- [Prodigy](https://prodi.gy/) - Active learning powered.
- [LightTag](https://lighttag.io/)
- [Scale AI](https://scale.com/)
- [UBIAI](https://ubiai.tools/)
- [tagtog](https://www.tagtog.net/)
- [Datasaur](https://datasaur.ai/)
- [Konfuzio](https://konfuzio.com/en/)

---

## Datasets

### Repositories

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)
- [nlp-datasets](https://github.com/niderhoff/nlp-datasets)
- [gensim-data](https://github.com/RaRe-Technologies/gensim-data)
- [tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp/)

### Pretraining

- [Common Crawl](https://commoncrawl.org/) · [The Pile](https://pile.eleuther.ai/) · [RedPajama](https://github.com/togethercomputer/RedPajama-Data) · [Dolma](https://github.com/allenai/dolma) · [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [The Stack](https://huggingface.co/datasets/bigcode/the-stack) (code)

### Instruction Tuning

- [FLAN Collection](https://github.com/google-research/FLAN) · [Natural Instructions](https://github.com/allenai/natural-instructions) · [P3](https://huggingface.co/datasets/bigscience/P3)
- [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) · [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) · [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) · [WizardLM](https://github.com/nlpxucan/WizardLM) · [Orca](https://arxiv.org/abs/2306.02707)

### Task-Specific

- **QA:** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) · [Natural Questions](https://ai.google.com/research/NaturalQuestions) · [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) · [HotpotQA](https://hotpotqa.github.io/)
- **Summarization:** [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) · [XSum](https://huggingface.co/datasets/xsum)
- **NLI:** [SNLI](https://nlp.stanford.edu/projects/snli/) · [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) · [ANLI](https://github.com/facebookresearch/anli)
- **NER:** [CoNLL-2003](https://huggingface.co/datasets/conll2003) · [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19) · [WikiANN](https://huggingface.co/datasets/wikiann)
- **Translation:** [WMT](https://www.statmt.org/wmt24/) · [OPUS](https://opus.nlpl.eu/) · [FLORES](https://github.com/facebookresearch/flores)

### Preference

- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) · [SHP](https://huggingface.co/datasets/stanfordnlp/SHP) · [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)

---

## Multilingual NLP

### Multilingual Models

- [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) (104 languages)
- [XLM-R](https://huggingface.co/xlm-roberta-base) (100 languages)
- [mT5](https://huggingface.co/google/mt5-base) (101 languages)
- [BLOOM](https://huggingface.co/bigscience/bloom) (46 languages)
- [Aya](https://huggingface.co/CohereForAI/aya-101) (101 languages)

### Translation

- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) (200 languages)
- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication)

### Multilingual Frameworks

- [UDPipe](https://github.com/ufal/udpipe) - Trainable pipeline for Universal Treebanks.
- [NLP-Cube](https://github.com/adobe/NLP-Cube) - Sentence splitting, tokenization, POS, parsing.
- [UralicNLP](https://github.com/mikahama/uralicNLP) - Uralic and other languages.
- [Stanza](https://stanfordnlp.github.io/stanza/)

---

<details>
<summary><strong>Language-Specific Resources</strong></summary>

### Chinese
**Libraries:** [jieba](https://github.com/fxsjy/jieba), [SnowNLP](https://github.com/isnowfy/snownlp), [HanLP](https://github.com/hankcs/HanLP), [FudanNLP](https://github.com/FudanNLP/fnlp)
**Models:** [Qwen](https://github.com/QwenLM/Qwen), [Yi](https://github.com/01-ai/Yi), [ChatGLM](https://github.com/THUDM/ChatGLM-6B), [Baichuan](https://github.com/baichuan-inc/Baichuan-7B)
**Resources:** [funNLP](https://github.com/fighting41love/funNLP)

### Japanese
**Libraries:** [MeCab](https://taku910.github.io/mecab/), [SudachiPy](https://github.com/WorksApplications/SudachiPy), [fugashi](https://github.com/polm/fugashi)
**Resources:** [awesome-japanese-nlp](https://github.com/taishi-i/awesome-japanese-nlp-resources)

### Korean
**Libraries:** [KoNLPy](http://konlpy.org), [Mecab-ko](https://eunjeon.blogspot.com/), [KoalaNLP](https://koalanlp.github.io/koalanlp/), [KoNLP](https://cran.r-project.org/package=KoNLP)
**Models:** [KoBERT](https://github.com/SKTBrain/KoBERT), [KoGPT](https://github.com/kakaobrain/kogpt), [KULLM](https://github.com/nlpai-lab/KULLM)
**Datasets:** [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus), [NSMC](https://github.com/e9t/nsmc/), [KorQuAD](https://korquad.github.io/), [Korean Parallel Corpora](https://github.com/j-min/korean-parallel-corpora)
**Tutorials:** [dsindex's blog](https://dsindex.github.io/), [Kangwon NLP course](http://cs.kangwon.ac.kr/~leeck/NLP/)

### Arabic
**Libraries:** [goarabic](https://github.com/01walid/goarabic), [jsastem](https://github.com/ejtaal/jsastem), [PyArabic](https://pypi.org/project/PyArabic/), [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools), [RFTokenizer](https://github.com/amir-zeldes/RFTokenizer)
**Models:** [AraBERT](https://github.com/aub-mind/arabert), [Jais](https://huggingface.co/inception-mbzuai/jais-13b)
**Datasets:** [LABR](https://github.com/mohamedadaly/labr), [Arabic Stopwords](https://github.com/mohataher/arabic-stop-words), [Multidomain Sentiment](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces)

### Hindi/Indic Languages
**Libraries:** [iNLTK](https://github.com/goru001/inltk), [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library), [Multi-Task DMA](https://github.com/Saurav0074/mt-dma)
**Models:** [IndicBERT](https://huggingface.co/ai4bharat/indic-bert), [MuRIL](https://huggingface.co/google/muril-base-cased), [Hindi2Vec](https://nirantk.com/hindi2vec/), [Sanskrit Albert](https://huggingface.co/surajp/albert-base-sanskrit)
**Datasets:** [Hindi Dependency Treebank](https://ltrc.iiit.ac.in/treebank_H2014/), [BBC News Hindi](https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1), [IIT Patna ABSA](https://github.com/pnisarg/ABSA)
**Resources:** [AI4Bharat](https://ai4bharat.org/)

### Thai
**Libraries:** [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp), [CutKum](https://github.com/pucktada/cutkum), [SynThai](https://github.com/KenjiroAI/SynThai), [JTCC](https://github.com/wittawatj/jtcc)
**Models:** [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
**Data:** [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm)

### Vietnamese
**Libraries:** [Underthesea](https://github.com/undertheseanlp/underthesea), [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP), [vn.vitk](https://github.com/phuonglh/vn.vitk), [pyvi](https://github.com/trungtv/pyvi)
**Models:** [PhoBERT](https://github.com/VinAIResearch/PhoBERT)
**Datasets:** [Vietnamese treebank](https://vlsp.hpda.vn/demo/?page=resources&lang=en), [BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf), [VIVOS](https://ailab.hcmus.edu.vn/vivos/), [ViText2SQL](https://github.com/VinAIResearch/ViText2SQL), [EVB Corpus](https://github.com/qhungngo/EVBCorpus)

### Persian
**Libraries:** [Hazm](https://github.com/roshan-research/hazm), [Parsivar](https://github.com/ICTRC/Parsivar), [Perke](https://github.com/AlirezaTheH/perke), [Perstem](https://github.com/jonsafari/perstem), [virastar](https://github.com/aziz/virastar)
**Models:** [ParsBERT](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
**Datasets:** [Bijankhan Corpus](https://dbrg.ut.ac.ir/بیژن%E2%80%8Cخان/), [Uppsala Persian Corpus](https://sites.google.com/site/mojganserajicom/home/upc), [LSCP](https://iasbs.ac.ir/~ansari/lscp/), [ArmanPersoNERCorpus](https://github.com/HaniehP/PersianNER), [PERLEX](http://farsbase.net/PERLEX.html)

### Indonesian
**Libraries:** [bahasa](https://github.com/kangfend/bahasa), [Indonesian Word Embedding](https://github.com/galuhsahid/indonesian-word-embedding)
**Models:** [IndoBERT](https://github.com/indobenchmark/indonlu)
**Datasets:** [IndoSum](https://github.com/kata-ai/indosum), [Wordnet-Bahasa](http://wn-msa.sourceforge.net/), [IndoNLU](https://github.com/indobenchmark/indonlu)

### Dutch
**Libraries:** [python-frog](https://github.com/proycon/python-frog), [Alpino](https://github.com/rug-compling/alpino), [SimpleNLG_NL](https://github.com/rfdj/SimpleNLG-NL), [Kaldi NL](https://github.com/opensource-spraakherkenning-nl/Kaldi_NL)
**Models:** [BERTje](https://github.com/wietsedv/bertje), [RobBERT](https://github.com/iPieter/RobBERT), [spaCy Dutch](https://spacy.io/models/nl)

### Spanish
**Libraries:** [spanlp](https://github.com/jfreddypuentes/spanlp)
**Models:** [BETO](https://github.com/dccuchile/beto)
**Embeddings:** [Spanish Word Embeddings](https://github.com/dccuchile/spanish-word-embeddings), [Spanish fastText](https://github.com/BotCenter/spanishWordEmbeddings), [Spanish sent2vec](https://github.com/BotCenter/spanishSent2Vec)
**Datasets:** [Columbian Political Speeches](https://github.com/dav009/LatinamericanTextResources), [Copenhagen Treebank](https://mbkromann.github.io/copenhagen-dependency-treebank/), [Spanish Billion Words](https://github.com/crscardellino/sbwce)

### German
- [German-NLP](https://github.com/adbar/German-NLP)

### Russian
- [Natasha](https://github.com/natasha/natasha), [pymorphy2](https://github.com/kmike/pymorphy2), [DeepPavlov](https://github.com/deeppavlov/DeepPavlov)

### Polish
- [Polish-NLP](https://github.com/ksopyla/awesome-nlp-polish)

### Portuguese
- [Portuguese-NLP](https://github.com/ajdavidl/Portuguese-NLP)

### Ukrainian
- [awesome-ukrainian-nlp](https://github.com/asivokon/awesome-ukrainian-nlp)
- [UkrainianLT](https://github.com/Helsinki-NLP/UkrainianLT)

### Hungarian
- [awesome-hungarian-nlp](https://github.com/oroszgy/awesome-hungarian-nlp)

### Danish
- [DaNLP](https://github.com/alexandrainst/danlp), [daner](https://github.com/ITUnlp/daner), [awesome-danish](https://github.com/fnielsen/awesome-danish)

### Urdu
- [Urduhack](https://github.com/urduhack/urduhack), [Urdu datasets](https://github.com/mirfan899/Urdu)

### Hebrew
- [NLPH_Resources](https://github.com/NLPH/NLPH_Resources)

### Ancient Languages
- [CLTK](https://github.com/cltk/cltk) - Classical Language Toolkit.

### Asian Languages (Thai, Lao, Chinese, Japanese, Korean)
- [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html) in ElasticSearch.

</details>

---

## Domain-Specific NLP

### Biomedical
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract), [BioBERT](https://github.com/dmis-lab/biobert), [BioGPT](https://github.com/microsoft/BioGPT), [ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [scispaCy](https://allenai.github.io/scispacy/), [MedCAT](https://github.com/CogStack/MedCAT)

### Legal
- [LegalBERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased), [Saul-7B](https://huggingface.co/Equall/Saul-7B-Base)

### Finance
- [FinBERT](https://github.com/ProsusAI/finBERT), [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)

### Scientific
- [SciBERT](https://github.com/allenai/scibert), [Galactica](https://huggingface.co/facebook/galactica-6.7b)
- [Semantic Scholar API](https://www.semanticscholar.org/product/api)

### Code
- [CodeBERT](https://github.com/microsoft/CodeBERT), [CodeT5](https://github.com/salesforce/CodeT5), [StarCoder](https://github.com/bigcode-project/starcoder)

---

## Essential Papers

**Classical NLP (1990s-2000s)**
- [A Maximum Entropy Approach to NLP](https://aclanthology.org/J96-1002/) (1996)
- [BLEU Score](https://aclanthology.org/P02-1040/) (2002)
- [Conditional Random Fields](https://repository.upenn.edu/cis_papers/159/) (2001)
- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/v3/blei03a.html) (2003)
- [A Unified Architecture for NLP](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf) (2008)

**Neural NLP (2013-2017)**
- [word2vec](https://arxiv.org/abs/1301.3781) (2013)
- [GloVe](https://aclanthology.org/D14-1162/) (2014)
- [Seq2Seq](https://arxiv.org/abs/1409.3215) (2014)
- [Attention](https://arxiv.org/abs/1409.0473) (2015)
- [ELMo](https://arxiv.org/abs/1802.05365) (2018)
- [ULMFiT](https://arxiv.org/abs/1801.06146) (2018)

**Transformer Era (2017-2021)**
- [Transformer](https://arxiv.org/abs/1706.03762) (2017)
- [BERT](https://arxiv.org/abs/1810.04805) (2018)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- [T5](https://arxiv.org/abs/1910.10683) (2019)
- [GPT-3](https://arxiv.org/abs/2005.14165) (2020)
- [Scaling Laws](https://arxiv.org/abs/2001.08361) (2020)
- [LoRA](https://arxiv.org/abs/2106.09685) (2021)

**LLM Era (2022-2023)**
- [InstructGPT](https://arxiv.org/abs/2203.02155) (2022)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903) (2022)
- [LLaMA](https://arxiv.org/abs/2302.13971) (2023)
- [DPO](https://arxiv.org/abs/2305.18290) (2023)
- [QLoRA](https://arxiv.org/abs/2305.14314) (2023)

**2024**
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (2024)
- [Mamba](https://arxiv.org/abs/2312.00752) (2024)
- [Llama 3](https://arxiv.org/abs/2407.21783) (2024)
- [Gemini 1.5](https://arxiv.org/abs/2403.05530) (2024)
- [Self-RAG](https://arxiv.org/abs/2310.11511) (2024)
- [Phi-3](https://arxiv.org/abs/2404.14219) (2024)

**2025-2026**
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2025)
- [Qwen2.5](https://arxiv.org/abs/2412.15115) (2025)
- [o1/o3 Reasoning](https://openai.com/index/learning-to-reason-with-llms/) (2025)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) (2026)

---

## Related Lists

- [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning)
- [awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
- [awesome-llm](https://github.com/Hannibal046/Awesome-LLM)

---

## Contributing

PRs welcome for new resources, broken link fixes, and updates.

---

## License

[CC0 1.0 Universal](./LICENSE)
