# awesome-nlp

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources dedicated to Natural Language Processing

_Please read the [contribution guidelines](contributing.md) before contributing. Please add your favourite NLP resource by raising a [pull request](https://github.com/keonkim/awesome-nlp/pulls)_

## Scope

This list covers natural language processing — linguistic analysis, multilingual tooling, classical and neural methods, datasets, and evaluation. Large language models are included only where they advance or evaluate a core NLP task or capability (tokenization, multilinguality, MT, summarization, NER, QA, factuality, probing, distillation). General-purpose chatbots, agent frameworks, prompt-template repositories, code-generation tools, and RAG application starter kits live in other lists — see [See Also](#see-also).

## Contents

* [Research Summaries and Trends](#research-summaries-and-trends)
* [Prominent NLP Research Labs](#prominent-nlp-research-labs)
* [Tutorials](#tutorials)
  * [Reading Content](#reading-content)
  * [Videos and Courses](#videos-and-online-courses)
  * [Books](#books)
* [Libraries](#libraries)
  * [Node.js](#node-js)
  * [Python](#python)
  * [C++](#c++)
  * [Java](#java)
  * [Kotlin](#kotlin)
  * [Scala](#scala)
  * [R](#R)
  * [Clojure](#clojure)
  * [Ruby](#ruby)
  * [Rust](#rust)
  * [NLP++](#NLP++)
  * [Julia](#julia)
* [Services](#services)
* [Annotation Tools](#annotation-tools)
* [Tasks and Methods](#tasks-and-methods)
  * [Text Embeddings](#text-embeddings)
  * [Tokenization, Morphology, and Segmentation](#tokenization-morphology-and-segmentation)
  * [POS Tagging and Dependency Parsing](#pos-tagging-and-dependency-parsing)
  * [Named Entity Recognition and Information Extraction](#named-entity-recognition-and-information-extraction)
  * [Coreference Resolution](#coreference-resolution)
  * [Text Classification and Sentiment Analysis](#text-classification-and-sentiment-analysis)
  * [Topic Modeling](#topic-modeling)
  * [Summarization](#summarization)
  * [Machine Translation](#machine-translation)
  * [Question Answering and Reading Comprehension](#question-answering-and-reading-comprehension)
  * [Information Extraction Beyond NER](#information-extraction-beyond-ner)
  * [Retrieval and Embeddings](#retrieval-and-embeddings)
  * [Speech and Text](#speech-and-text)
* [Datasets](#datasets)
* [Multilingual NLP Frameworks](#multilingual-nlp-frameworks)
* [Language Models for NLP](#language-models-for-nlp)
  * [Pretraining and Adaptation](#pretraining-and-adaptation)
  * [Multilingual and Cross-Lingual Models](#multilingual-and-cross-lingual-models)
  * [Evaluation and Benchmarks](#evaluation-and-benchmarks)
  * [Reasoning and Test-Time Compute](#reasoning-and-test-time-compute)
  * [Long Context and Alternative Architectures](#long-context-and-alternative-architectures)
  * [Factuality, Hallucination, Calibration](#factuality-hallucination-calibration)
  * [Probing and Interpretability](#probing-and-interpretability)
  * [Efficient and Small Language Models](#efficient-and-small-language-models)
  * [Instruction Tuning and Preference Optimization](#instruction-tuning-and-preference-optimization)
  * [Bias, Fairness, Safety in NLP](#bias-fairness-safety-in-nlp)
* [NLP per Language](#nlp-per-language)
  * [NLP in Arabic](#nlp-in-arabic)
  * [NLP in Chinese](#nlp-in-chinese)
  * [NLP in Danish](#nlp-in-danish)
  * [NLP in Dutch](#nlp-in-dutch)
  * [NLP in German](#nlp-in-german)
  * [NLP in Hungarian](#nlp-in-hungarian)
  * [NLP in Indic Languages](#nlp-in-indic-languages)
  * [NLP in Indonesian](#nlp-in-indonesian)
  * [NLP in Korean](#nlp-in-korean)
  * [NLP in Persian](#nlp-in-persian)
  * [NLP in Polish](#nlp-in-polish)
  * [NLP in Portuguese](#nlp-in-portuguese)
  * [NLP in Spanish](#nlp-in-spanish)
  * [NLP in Thai](#nlp-in-thai)
  * [NLP in Ukrainian](#nlp-in-ukrainian)
  * [NLP in Urdu](#nlp-in-urdu)
  * [NLP in Vietnamese](#nlp-in-vietnamese)
  * [Other Languages](#other-languages)
* [See Also](#see-also)
* [Citation](#citation)

## Research Summaries and Trends

Where to follow current NLP research:

* [ACL Anthology](https://aclanthology.org/) - canonical archive of papers from ACL, EMNLP, NAACL, EACL, COLING, and related venues.
* [NLP-Progress](https://nlpprogress.com/) - tracks state-of-the-art results across common NLP tasks and datasets.
* [Papers With Code: NLP](https://paperswithcode.com/area/natural-language-processing) - papers, benchmarks, and leaderboards for NLP tasks.
* [Sebastian Ruder's newsletter](https://newsletter.ruder.io/) - regular roundups of NLP research and trends.
* [ACL Rolling Review](https://aclrollingreview.org/) - the rolling review process feeding ACL-affiliated venues.
* [The Gradient](https://thegradient.pub/) - long-form essays on ML and NLP research.
* [Visual NLP Paper Summaries](https://amitness.com/categories/#nlp) - illustrated summaries of recent papers.

### Historical highlights

* [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/) - 2018 essay on the rise of pretrained language models.
* [Survey of the State of the Art in Natural Language Generation](https://arxiv.org/abs/1703.09902) - 2017 NLG survey.
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) - canonical visual explanations.

## Prominent NLP Research Labs
[Back to Top](#contents)

* [The Berkeley NLP Group](http://nlp.cs.berkeley.edu/index.shtml) - Notable contributions include a tool to reconstruct long dead languages, referenced [here](https://www.bbc.com/news/science-environment-21427896) and by taking corpora from 637 languages currently spoken in Asia and the Pacific and recreating their descendant.
* [Language Technologies Institute, Carnegie Mellon University](http://www.cs.cmu.edu/~nasmith/nlp-cl.html) - Notable projects include [Avenue Project](http://www.cs.cmu.edu/~avenue/), a syntax driven machine translation system for endangered languages like Quechua and Aymara and previously, [Noah's Ark](http://www.cs.cmu.edu/~ark/) which created [AQMAR](http://www.cs.cmu.edu/~ark/AQMAR/) to improve NLP tools for Arabic.
* [NLP research group, Columbia University](http://www1.cs.columbia.edu/nlp/index.cgi) - Responsible for creating BOLT ( interactive error handling for speech translation systems) and an un-named project to characterize laughter in dialogue.
* [The Center or Language and Speech Processing, John Hopkins University](http://clsp.jhu.edu/) - Recently in the news for developing speech recognition software to create a diagnostic test or Parkinson's Disease, [here](https://www.clsp.jhu.edu/2019/03/27/speech-recognition-software-and-machine-learning-tools-are-being-used-to-create-diagnostic-test-for-parkinsons-disease/#.XNFqrIkzYdU).
* [Computational Linguistics and Information Processing Group, University of Maryland](https://wiki.umiacs.umd.edu/clip/index.php/Main_Page) - Notable contributions include [Human-Computer Cooperation or Word-by-Word Question Answering](http://www.umiacs.umd.edu/~jbg/projects/IIS-1652666) and modeling development of phonetic representations. 
* [Penn Natural Language Processing, University of Pennsylvania](https://nlp.cis.upenn.edu/) - famous for creating the [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) and the [Penn Discourse Treebank](https://www.cis.upenn.edu/~pdtb/).
* [The Stanford Nautral Language Processing Group](https://nlp.stanford.edu/)- One of the top NLP research labs in the world, notable for creating [Stanford CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) and their [coreference resolution system](https://nlp.stanford.edu/software/dcoref.shtml)


## Tutorials
[Back to Top](#contents)

### Reading Content

General Machine Learning

* [Machine Learning 101](https://docs.google.com/presentation/d/1kSuQyW5DTnkVaZEjGYCkfOxvzCqGEFzWBy4e9Uedd9k/edit?usp=sharing) from Google's Senior Creative Engineer explains Machine Learning for engineer's and executives alike
* [AI Playbook](https://aiplaybook.a16z.com/) - a16z AI playbook is a great link to forward to your managers or content for your presentations
* [Sebastian Ruder's Newsletter](https://newsletter.ruder.io/) for commentary on the best of NLP research.
* [How To Label Data](https://www.lighttag.io/how-to-label-data/) guide to managing larger linguistic annotation projects
* [Depends on the Definition](https://www.depends-on-the-definition.com/) collection of blog posts covering a wide array of NLP topics with detailed implementation

Introductions and Guides to NLP

* [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [NLP in Python](http://github.com/NirantK/nlp-python-deep-learning) - Collection of Github notebooks
* [Natural Language Processing: An Introduction](https://academic.oup.com/jamia/article/18/5/544/829676) - Oxford
* [NLP from Scratch with PyTorch](https://pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html)
* [Hands-On NLTK Tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) - NLTK Tutorials, Jupyter notebooks
* [Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit](https://www.nltk.org/book/) - An online and print book introducing NLP concepts using NLTK. The book's authors also wrote the NLTK library.
* [Train a new language model from scratch](https://huggingface.co/blog/how-to-train) - Hugging Face 🤗
* [Advanced NLP with spaCy](https://course.spacy.io/en/) - Free online course covering text processing, large-scale data analysis, processing pipelines, and training neural network models for custom NLP tasks.
* [Kaggle NLP Learning Guide](https://www.kaggle.com/learn-guide/natural-language-processing) - Beginner-friendly tutorials including getting started guides, deep learning for NLP, and visual explanations of techniques like BERT, GloVe, and TF-IDF.

Blogs and Newsletters

* [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/) and [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Natural Language Processing](https://nlpers.blogspot.com/) by Hal Daumé III
* [arXiv: Natural Language Processing (Almost) from Scratch](https://arxiv.org/pdf/1103.0398.pdf)
* [Karpathy's The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
* [Machine Learning Mastery: Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing)
* [Visual NLP Paper Summaries](https://amitness.com/categories/#nlp)

### Videos and Online Courses
[Back to Top](#contents)

* [Advanced Natural Language Processing](https://people.cs.umass.edu/~miyyer/cs685_f20/) - CS 685, UMass Amherst CS
* [Deep Natural Language Processing](https://github.com/oxford-cs-deepnlp-2017/lectures) - Lectures series from Oxford
* [Deep Learning for Natural Language Processing (cs224-n)](https://web.stanford.edu/class/cs224n/) - Richard Socher and Christopher Manning's Stanford Course
* [Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/) - Carnegie Mellon Language Technology Institute there
* [Deep NLP Course](https://github.com/yandexdataschool/nlp_course) by Yandex Data School, covering important ideas from text embedding to machine translation including sequence modeling, language models and so on.
* [fast.ai Code-First Intro to Natural Language Processing](https://www.fast.ai/2019/07/08/fastai-nlp/) - This covers a blend of traditional NLP topics (including regex, SVD, naive bayes, tokenization) and recent neural network approaches (including RNNs, seq2seq, GRUs, and the Transformer), as well as addressing urgent ethical issues, such as bias and disinformation. Find the Jupyter Notebooks [here](https://github.com/fastai/course-nlp)
* [Machine Learning University - Accelerated Natural Language Processing](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw) - Lectures go from introduction to NLP and text processing to Recurrent Neural Networks and Transformers.
Material can be found [here](https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp).
* [Applied Natural Language Processing](https://www.youtube.com/playlist?list=PLH-xYrxjfO2WyR3pOAB006CYMhNt4wTqp)- Lecture series from IIT Madras taking from the basics all the way to autoencoders and everything. The github notebooks for this course are also available [here](https://github.com/Ramaseshanr/anlp)
* [DeepLearning.AI Natural Language Processing Specialization](https://www.deeplearning.ai/courses/natural-language-processing-specialization/) - 4-course program covering sentiment analysis, word embeddings, RNNs, LSTMs, attention mechanisms, and Transformer models like BERT and T5 for tasks including machine translation and summarization.
* [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) - end-to-end course on building language models, including data, tokenization, training, and evaluation.
* [Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/) - seminar series with guest lectures from authors of recent transformer and NLP research.
* [Cohere LLM University](https://cohere.com/llmu) - free course on LLMs, embeddings, semantic search, and NLP applications.
* [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) - hands-on NLP with Transformers, Datasets, and Tokenizers libraries.


### Books

* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - free, by Prof. Dan Jurafsy
* [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class) - free, NLP notes by Dr. Jacob Eisenstein at GeorgiaTech
* [NLP with PyTorch](https://github.com/joosthub/PyTorchNLPBook) - Brian & Delip Rao
* [Text Mining in R](https://www.tidytextmining.com)
* [Natural Language Processing with Python](https://www.nltk.org/book/)
* [Practical Natural Language Processing](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/)
* [Natural Language Processing with Spark NLP](https://www.oreilly.com/library/view/natural-language-processing/9781492047759/)
* [Deep Learning for Natural Language Processing](https://www.manning.com/books/deep-learning-for-natural-language-processing) by Stephan Raaijmakers
* [Real-World Natural Language Processing](https://www.manning.com/books/real-world-natural-language-processing) - by Masato Hagiwara
* [Natural Language Processing in Action, Second Edition](https://www.manning.com/books/natural-language-processing-in-action-second-edition) - by Hobson Lane and Maria Dyshel
* [Transformers in Action](https://www.manning.com/books/transformers-in-action) - by Nicole Koenigstein
* [The Math Behind Artificial Intelligence](https://www.freecodecamp.org/news/the-math-behind-artificial-intelligence-book) - bt Tiago MOnteiro | A free FreeCodeCamp book teaching the math behind AI in plain English from an engineering point of view. It covers linear algebra, calculus, probability & statistics, and optimization theory with analogies, real-life applications, and Python code examples.
  
## Libraries

[Back to Top](#contents)

* <a id="node-js">**Node.js and Javascript** - Node.js Libaries for NLP</a> | [Back to Top](#contents)
  * [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library
  * [Knwl.js](https://github.com/benhmoore/Knwl.js) - A Natural Language Processor in JS
  * [Retext](https://github.com/retextjs/retext) - Extensible system for analyzing and manipulating natural language
  * [NLP Compromise](https://github.com/spencermountain/compromise) - Natural Language processing in the browser
  * [Natural](https://github.com/NaturalNode/natural) - general natural language facilities for node
  * [Poplar](https://github.com/synyi/poplar) - A web-based annotation tool for natural language processing (NLP)
  * [NLP.js](https://github.com/axa-group/nlp.js) - An NLP library for building bots
  * [node-question-answering](https://github.com/huggingface/node-question-answering) - Fast and production-ready question answering w/ DistilBERT in Node.js

* <a id="python"> **Python** - Python NLP Libraries</a> | [Back to Top](#contents)
  - [sentimental-onix](https://github.com/sloev/sentimental-onix) Sentiment models for spacy using onnx
  - [TextAttack](https://github.com/QData/TextAttack) - Adversarial attacks, adversarial training, and data augmentation in NLP
  - [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of [Natural Language Toolkit (NLTK)](https://www.nltk.org/) and [Pattern](https://github.com/clips/pattern), and plays nicely with both :+1:
  - [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython :+1:
    - [textacy](https://github.com/chartbeat-labs/textacy) - Higher level NLP built on spaCy
  - [gensim](https://radimrehurek.com/gensim/index.html) - Python library to conduct unsupervised semantic modelling from plain text :+1:
  - [scattertext](https://github.com/JasonKessler/scattertext) - Python library to produce d3 visualizations of how language differs between corpora
  - [GluonNLP](https://github.com/dmlc/gluon-nlp) *(archived)* - A deep learning toolkit for NLP, built on MXNet/Gluon.
  - [AllenNLP](https://github.com/allenai/allennlp) *(archived)* - An NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
  - [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP research toolkit designed to support rapid prototyping with better data loaders, word vector loaders, neural network layer representations, common NLP metrics such as BLEU
  - [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
  - [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python, handles some specific formats like ARPA language models, Moses phrasetables, GIZA++ alignments.
  - [foliapy](https://github.com/proycon/foliapy) - Python library for working with [FoLiA](https://proycon.github.io/folia/), an XML format for linguistic annotation.
  - [PySS3](https://github.com/sergioburdisso/pyss3) - Python package implementing the SS3 white-box text classifier; ships with interactive visualization tools that explain predictions.
  - [jPTDP](https://github.com/datquocnguyen/jPTDP) - A toolkit for joint part-of-speech (POS) tagging and dependency parsing. jPTDP provides pre-trained models for 40+ languages.
  - [BigARTM](https://github.com/bigartm/bigartm) - a fast library for topic modelling
  - [Snips NLU](https://github.com/snipsco/snips-nlu) - A production ready library for intent parsing
  - [Chazutsu](https://github.com/chakki-works/chazutsu) - A library for downloading&parsing standard NLP research datasets
  - [Word Forms](https://github.com/gutfeeling/word_forms) - Word forms can accurately generate all possible forms of an English word
  - [Multilingual Latent Dirichlet Allocation (LDA)](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA) - A multilingual and extensible document clustering pipeline
  - [Natural Language Toolkit (NLTK)](https://www.nltk.org/) - A library containing a wide variety of NLP functionality, supporting over 50 corpora.
  - [NLP Architect](https://github.com/NervanaSystems/nlp-architect) - A library for exploring the state-of-the-art deep learning topologies and techniques for NLP and NLU
  - [Flair](https://github.com/zalandoresearch/flair) - A very simple framework for state-of-the-art multilingual NLP built on PyTorch. Includes BERT, ELMo and Flair embeddings.
  - [Kashgari](https://github.com/BrikerMan/Kashgari) - Simple, Keras-powered multilingual NLP framework, allows you to build your models in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS) and text classification tasks. Includes BERT and word2vec embedding.
  - [FARM](https://github.com/deepset-ai/FARM) - Fast & easy transfer learning for NLP. Harvesting language models for the industry. Focus on Question Answering.
  - [Haystack](https://github.com/deepset-ai/haystack) - End-to-end Python framework for building natural language search interfaces to data. Leverages Transformers and the State-of-the-Art of NLP. Supports DPR, Elasticsearch, HuggingFace’s Modelhub, and much more!
  - [Rita DSL](https://github.com/zaibacu/rita-dsl) - a DSL, loosely based on [RUTA on Apache UIMA](https://uima.apache.org/ruta.html). Allows to define language patterns (rule-based NLP) which are then translated into [spaCy](https://spacy.io/), or if you prefer less features and lightweight - regex patterns.
  - [Transformers](https://github.com/huggingface/transformers) - Natural Language Processing for TensorFlow 2.0 and PyTorch.
  - [Tokenizers](https://github.com/huggingface/tokenizers) - Tokenizers optimized for Research and Production.
  - [fairSeq](https://github.com/pytorch/fairseq) Facebook AI Research implementations of SOTA seq2seq models in Pytorch. 
  - [corex_topic](https://github.com/gregversteeg/corex_topic) - Hierarchical Topic Modeling with Minimal Domain Knowledge
  - [Sockeye](https://github.com/awslabs/sockeye) - Neural Machine Translation (NMT) toolkit that powers Amazon Translate.
  - [DL Translate](https://github.com/xhlulu/dl-translate) - A deep learning-based translation library for 50 languages, built on `transformers` and Facebook's mBART Large.
  - [Jury](https://github.com/obss/jury) - Evaluation of NLP model outputs offering various automated metrics.
  - [python-ucto](https://github.com/proycon/python-ucto) - Unicode-aware regular-expression based tokenizer for various languages. Python binding to C++ library, supports [FoLiA format](https://proycon.github.io/folia).
  - [Pearmut](https://github.com/zouharvi/pearmut) - Human annotation tool for multilingual NLP tasks, such as machine translation.
  - [Stanza](https://github.com/stanfordnlp/stanza) - Stanford NLP's Python toolkit for tokenization, POS, lemma, dependency parsing, and NER across 70+ languages.
  - [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) - sentence/document embeddings, semantic search, and re-ranking; current standard for retrieval-style NLP.
  - [Argilla](https://github.com/argilla-io/argilla) - open-source data annotation and feedback collection platform for LLM and NLP datasets.
  - [HuggingFace Datasets](https://github.com/huggingface/datasets) - standardized loaders and processing for thousands of NLP datasets.
  - [HuggingFace Evaluate](https://github.com/huggingface/evaluate) - reference implementations for NLP metrics.
  - [sacrebleu](https://github.com/mjpost/sacrebleu) - reproducible BLEU/chrF/TER scoring for machine translation.
  - [COMET](https://github.com/Unbabel/COMET) - learned MT metrics, current de-facto standard.
  - [LangTest](https://github.com/JohnSnowLabs/langtest) - 60+ test types for NLP model robustness, bias, and fairness.

- <a id="c++">**C++** - C++ Libraries</a> | [Back to Top](#contents)
  - [InsNet](https://github.com/chncwang/InsNet) - A neural network library for building instance-dependent NLP models with padding-free dynamic batching.
  - [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
  - [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
  - [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data.
  - [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  - [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  - [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
  - [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia/)
  - [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
  - [MeTA](https://github.com/meta-toolkit/meta) - ModErn Text Analysis: a C++ data sciences toolkit for mining big text data.
  - [Mecab (Japanese)](https://taku910.github.io/mecab/)
  - [Moses](http://statmt.org/moses/)
  - [StarSpace](https://github.com/facebookresearch/StarSpace) - a library from Facebook for creating embeddings of word-level, paragraph-level, document-level and for text classification
  - [QSMM](http://qsmm.org) - adaptive probabilistic top-down and bottom-up parsers

- <a id="java">**Java** - Java NLP Libraries</a> | [Back to Top](#contents)
  - [Stanford NLP](https://nlp.stanford.edu/software/index.shtml)
  - [OpenNLP](https://opennlp.apache.org/)
  - [NLP4J](https://emorynlp.github.io/nlp4j/)
  - [Word2vec in Java](https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec)
  - [ReVerb](https://github.com/knowitall/reverb/) Web-Scale Open Information Extraction
  - [OpenRegex](https://github.com/knowitall/openregex) An efficient and flexible token-based regular expression language and engine.
  - [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - Core libraries developed in the U of Illinois' Cognitive Computation Group.
  - [MALLET](http://mallet.cs.umass.edu/) - MAchine Learning for LanguagE Toolkit - package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text.
  - [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - A robust POS tagging toolkit available (in both Java & Python) together with pre-trained models for 40+ languages.

- <a id="kotlin">**Kotlin** - Kotlin NLP Libraries</a> | [Back to Top](#contents)
  - [Lingua](https://github.com/pemistahl/lingua/) A language detection library for Kotlin and Java, suitable for long and short text alike
  - [Kotidgy](https://github.com/meiblorn/kotidgy) — an index-based text data generator written in Kotlin

- <a id="scala">**Scala** - Scala NLP Libraries</a> | [Back to Top](#contents)
  - [Saul](https://github.com/CogComp/saul) - Library for developing NLP systems, including built in modules like SRL, POS, etc.
  - [ATR4S](https://github.com/ispras/atr4s) - Toolkit with state-of-the-art [automatic term recognition](https://en.wikipedia.org/wiki/Terminology_extraction) methods.
  - [tm](https://github.com/ispras/tm) - Implementation of topic modeling based on regularized multilingual [PLSA](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis).
  - [word2vec-scala](https://github.com/Refefer/word2vec-scala) - Scala interface to word2vec model; includes operations on vectors like word-distance and word-analogy.
  - [Epic](https://github.com/dlwh/epic) - Epic is a high performance statistical parser written in Scala, along with a framework for building complex structured prediction models.
  - [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) - Spark NLP is a natural language processing library built on top of Apache Spark ML that provides simple, performant & accurate NLP annotations for machine learning pipelines that scale easily in a distributed environment.

- <a id="R">**R** - R NLP Libraries</a> | [Back to Top](#contents)
  - [text2vec](https://github.com/dselivanov/text2vec) - Fast vectorization, topic modeling, distances and GloVe word embeddings in R.
  - [wordVectors](https://github.com/bmschmidt/wordVectors) - An R package for creating and exploring word2vec and other word embedding models
  - [RMallet](https://github.com/mimno/RMallet) - R package to interface with the Java machine learning tool MALLET
  - [dfr-browser](https://github.com/agoldst/dfr-browser) - Creates d3 visualizations for browsing topic models of text in a web browser.
  - [dfrtopics](https://github.com/agoldst/dfrtopics) - R package for exploring topic models of text.
  - [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - Sentiment Classification using Word Sense Disambiguation and WordNet Reader
  - [jProcessing](https://github.com/kevincobain2000/jProcessing) - Japanese Natural Langauge Processing Libraries, with Japanese sentiment classification
  - [corporaexplorer](https://kgjerde.github.io/corporaexplorer/) - An R package for dynamic exploration of text collections
  - [tidytext](https://github.com/juliasilge/tidytext) - Text mining using tidy tools
  - [spacyr](https://github.com/quanteda/spacyr) - R wrapper to spaCy NLP
  - [CRAN Task View: Natural Language Processing](https://github.com/cran-task-views/NaturalLanguageProcessing/)

- <a id="clojure">**Clojure**</a> | [Back to Top](#contents)
  - [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
  - [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript
  - [postagga](https://github.com/fekr/postagga) - A library to parse natural language in Clojure and ClojureScript

- <a id="ruby">**Ruby**</a> | [Back to Top](#contents)
  - Kevin Dias's [A collection of Natural Language Processing (NLP) Ruby libraries, tools and software](https://github.com/diasks2/ruby-nlp)
  - [Practical Natural Language Processing done in Ruby](https://github.com/arbox/nlp-with-ruby)

- <a id="rust">**Rust**</a> | [Back to Top](#contents)
  - [whatlang](https://github.com/greyblake/whatlang-rs) — Natural language recognition library based on trigrams
  - [rust-bert](https://github.com/guillaume-be/rust-bert) - Ready-to-use NLP pipelines and Transformer-based models
  - [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) *(archived — Snips was discontinued)* - A production ready library for intent parsing

- <a id="NLP++">**NLP++** - NLP++ Language</a> | [Back to Top](#contents)
  - [VSCode Language Extension](https://marketplace.visualstudio.com/items?itemName=dehilster.nlp) - NLP++ Language Extension for VSCode
  - [nlp-engine](https://github.com/VisualText/nlp-engine) - NLP++ engine to run NLP++ code on Linux including a full English parser
  - [VisualText](http://visualtext.org) - Homepage for the NLP++ Language
  - [NLP++ Wiki](http://wiki.naturalphilosophy.org/index.php?title=NLP%2B%2B) - Wiki entry for the NLP++ language

- <a id="julia">**Julia**</a> | [Back to Top](#contents)
  - [CorpusLoaders](https://github.com/JuliaText/CorpusLoaders.jl) - A variety of loaders for various NLP corpora
  - [Languages](https://github.com/JuliaText/Languages.jl) - A package for working with human languages
  - [TextAnalysis](https://github.com/JuliaText/TextAnalysis.jl) - Julia package for text analysis
  - [TextModels](https://github.com/JuliaText/TextModels.jl) - Neural Network based models for Natural Language Processing
  - [WordTokenizers](https://github.com/JuliaText/WordTokenizers.jl) - High performance tokenizers for natural language processing and other related tasks
  - [Word2Vec](https://github.com/JuliaText/Word2Vec.jl) - Julia interface to word2vec

### Services

NLP as API with higher level functionality such as NER, Topic tagging and so on | [Back to Top](#contents)

- [Wit-ai](https://github.com/wit-ai/wit) - Natural Language Interface for apps and devices
- [IBM Watson's Natural Language Understanding](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs) - API and Github demo
- [Amazon Comprehend](https://aws.amazon.com/comprehend/) - NLP and ML suite covers most common tasks like NER, tagging, and sentiment analysis
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language/) - Syntax Analysis, NER, Sentiment Analysis, and Content tagging in atleast 9 languages include English and Chinese (Simplified and Traditional).
- [ParallelDots](https://www.paralleldots.com/text-analysis-apis) - High level Text Analysis API Service ranging from Sentiment Analysis to Intent Analysis
- [Microsoft Cognitive Service](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
- [TextRazor](https://www.textrazor.com/)
- [Rosette](https://www.rosette.com/)
- [Textalytic](https://www.textalytic.com) - Natural Language Processing in the Browser with sentiment analysis, named entity extraction, POS tagging, word frequencies, topic modeling, word clouds, and more
- [NLP Cloud](https://nlpcloud.io) - SpaCy NLP models (custom and pre-trained ones) served through a RESTful API for named entity recognition (NER), POS tagging, and more.
- [Cloudmersive](https://cloudmersive.com/nlp-api) - Unified and free NLP APIs that perform actions such as speech tagging, text rephrasing, language translation/detection, and sentence parsing
- [Not Human Search](https://nothumansearch.ai) - Search engine and MCP server for discovering agent-ready NLP and AI tools, with a REST API and agentic readiness scoring based on llms.txt, OpenAPI, and MCP support

### Annotation Tools

- [GATE](https://gate.ac.uk/overview.html) - General Architecture and Text Engineering is 15+ years old, free and open source
- [Anafora](https://github.com/weitechen/anafora) is free and open source, web-based raw text annotation tool
- [brat](https://brat.nlplab.org/) - brat rapid annotation tool is an online environment for collaborative text annotation
- [doccano](https://github.com/chakki-works/doccano) - doccano is free, open-source, and provides annotation features for text classification, sequence labeling and sequence to sequence
- [INCEpTION](https://inception-project.github.io) - A semantic annotation platform offering intelligent assistance and knowledge management
- [prodigy](https://prodi.gy/) is an annotation tool powered by active learning, costs $
- [LightTag](https://lighttag.io) - Hosted and managed text annotation tool for teams, costs $
- [rstWeb](https://corpling.uis.georgetown.edu/rstweb/info/) - open source local or online tool for discourse tree annotations
- [GitDox](https://corpling.uis.georgetown.edu/gitdox/) - open source server annotation tool with GitHub version control and validation for XML data and collaborative spreadsheet grids
- [Datasaur](https://datasaur.ai/) support various NLP tasks for individual or teams, freemium based
- [Konfuzio](https://konfuzio.com/en/) - team-first hosted and on-prem text, image and PDF annotation tool powered by active learning, freemium based, costs $
- [UBIAI](https://ubiai.tools/) - Easy-to-use text annotation tool for teams with most comprehensive auto-annotation features. Supports NER, relations and document classification as well as OCR annotation for invoice labeling, costs $
- [Shoonya](https://github.com/AI4Bharat/Shoonya-Backend) - Shoonya is free and open source data annotation platform with wide varities of organization and workspace level management system. Shoonya is data agnostic, can be used by teams to annotate data with various level of verification stages at scale.
- [Annotation Lab](https://www.johnsnowlabs.com/annotation-lab/) - Free End-to-End No-Code platform for text annotation and DL model training/tuning. Out-of-the-box support for Named Entity Recognition, Classification, Relation extraction and Assertion Status Spark NLP models. Unlimited support for users, teams, projects, documents. Not FOSS. 
- [FLAT](https://github.com/proycon/flat) - FLAT is a web-based linguistic annotation environment based around the [FoLiA format](http://proycon.github.io/folia), a rich XML-based format for linguistic annotation. Free and open source.
- [Argilla](https://github.com/argilla-io/argilla) - open-source platform for collecting human feedback, building NLP and LLM datasets, and curating preference data.
- [Label Studio](https://github.com/HumanSignal/label-studio) - open-core multi-modal labeling platform; widely used for NLP labeling.


## Tasks and Methods

NLP tasks organized by linguistic problem. Each subsection lists foundational/classical work first, then neural approaches, then LLM-based methods where relevant. For modern LM-specific research (pretraining, evaluation, retrieval, reasoning, etc.) see [Language Models for NLP](#language-models-for-nlp).

### Text Embeddings

[Back to Top](#contents)

Static word embeddings (foundational):

- [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - [implementation](https://code.google.com/archive/p/word2vec/) - [explainer blog](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) - [explainer blog](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/)
- [fastText](https://arxiv.org/abs/1607.04606) - [implementation](https://github.com/facebookresearch/fastText); subword n-grams handle OOV well, still useful for low-resource languages.
- [sense2vec](https://arxiv.org/abs/1511.06388) - word sense disambiguation.
- [Paragraph Vectors / doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

Contextual embeddings:

- [ELMo](https://arxiv.org/abs/1802.05365) - deep contextualized word representations.
- [CoVe](https://arxiv.org/abs/1708.00107) - contextualized vectors learned from MT.
- [ULMFiT](https://arxiv.org/abs/1801.06146) - language-model fine-tuning for text classification.
- [InferSent](https://arxiv.org/abs/1705.02364) - sentence representations from NLI.

Modern sentence and document embeddings: see [Retrieval for NLP](#retrieval-for-nlp) (Sentence-Transformers, E5, BGE-M3, Nomic, GritLM) and [MTEB](https://github.com/embeddings-benchmark/mteb) for current leaderboards.

### Tokenization, Morphology, and Segmentation

[Back to Top](#contents)

- [SentencePiece](https://github.com/google/sentencepiece) - language-agnostic subword tokenization.
- [BPE](https://arxiv.org/abs/1508.07909) and [Unigram LM](https://arxiv.org/abs/1804.10959) - the two dominant subword schemes.
- [Stanza](https://github.com/stanfordnlp/stanza) - tokenization, lemma, and morphology for 70+ languages.
- [UDPipe](https://github.com/ufal/udpipe) - tokenization, tagging, lemmatization, parsing for Universal Dependencies.
- [Morfessor](https://github.com/aalto-speech/morfessor) - unsupervised morphological segmentation.
Tokenizer research and architecture (also see [Language Models](#language-models-for-nlp)):

- [Byte-Pair Encoding (Sennrich et al.)](https://arxiv.org/abs/1508.07909) - subword units for neural MT; foundation of modern tokenizers.
- [SentencePiece](https://github.com/google/sentencepiece) - language-agnostic subword tokenization (BPE and Unigram).
- [Tokenizers](https://github.com/huggingface/tokenizers) - fast Rust implementations of BPE, WordPiece, Unigram.
- [ByT5](https://arxiv.org/abs/2105.13626) - tokenizer-free byte-level model.
- [CANINE](https://arxiv.org/abs/2103.06874) - tokenization-free encoder operating on Unicode characters.
- [How Good is Your Tokenizer?](https://arxiv.org/abs/2012.15613) - tokenizer fairness across languages.
- [Byte Latent Transformer (BLT)](https://arxiv.org/abs/2412.09871) (Meta, 2024) - dynamic byte-level patching that matches BPE-tokenized models at scale; revives the tokenizer-free direction.
- [SuperBPE](https://arxiv.org/abs/2503.13423) (2025) - superword tokenization that improves on BPE for downstream tasks.
- [Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975) (ICML 2025) - decouples input and output vocabularies; shows a log-linear relationship between input vocabulary size and training loss, scaling vocabulary independently of model size.
- [Foundations of Tokenization](https://arxiv.org/abs/2407.11606) (ICLR 2025) - first formal unified framework for tokenizer models using stochastic-map category theory; establishes conditions for statistical consistency.
- [The Token Tax: Systematic Bias in Multilingual Tokenization](https://arxiv.org/abs/2509.05486) (2025) - quantifies how tokenization fertility predicts model accuracy across languages, exposing structural cost penalties for morphologically complex and low-resource languages.
- [Reducing Tokenization Premiums for Low-Resource Languages](https://arxiv.org/abs/2601.13328) (2026) - post-hoc vocabulary additions that coalesce multi-token character sequences for low-resource languages, reducing inference cost without retraining.

### POS Tagging and Dependency Parsing

[Back to Top](#contents)

- [Universal Dependencies](https://universaldependencies.org/) - cross-linguistically consistent treebanks, 100+ languages.
- [spaCy](https://spacy.io/) and [Stanza](https://github.com/stanfordnlp/stanza) - production parsers across many languages.
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734) - foundational neural parsing architecture.
- [Trankit](https://github.com/nlp-uoregon/trankit) - light-weight transformer-based multilingual NLP toolkit.
- [Self-Attentive Constituency Parsing (Kitaev & Klein)](https://arxiv.org/abs/1805.01052) - strong neural constituency parser.

### Named Entity Recognition and Information Extraction

[Back to Top](#contents)

Foundational and neural:

- [CoNLL-2003 NER](https://www.aclweb.org/anthology/W03-0419/) - canonical English NER benchmark.
- [Neural Architectures for NER (Lample et al.)](https://arxiv.org/abs/1603.01360) - BiLSTM-CRF, the long-time go-to NER architecture.
- [Flair](https://github.com/flairNLP/flair) - contextual string embeddings, strong NER across languages.
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) - production-ready.

Open and instruction-following IE:

- [Universal NER](https://arxiv.org/abs/2308.03279) - instruction-tuned LM for open-set NER across languages.
- [GLiNER](https://arxiv.org/abs/2311.08526) (2023) - small, generalist NER model that handles arbitrary entity types at inference.
- [GoLLIE](https://arxiv.org/abs/2310.03668) - guideline-following information extraction with LMs.
- [REBEL](https://github.com/Babelscape/rebel) - end-to-end relation extraction as seq2seq.

LLM-based:

- [GPT-NER](https://arxiv.org/abs/2304.10428) - LLMs for named entity recognition.
- [Can LLMs Replace Sentence-Level NER?](https://arxiv.org/abs/2402.10573) (2024) - cost-quality tradeoffs.
- [Generative NER in the Era of LLMs](https://arxiv.org/abs/2601.17898) (2026) - eight open LLMs across four NER benchmarks; PEFT with structured outputs matches encoder-based NER.

### Coreference Resolution

[Back to Top](#contents)

- [End-to-End Neural Coreference (Lee et al.)](https://arxiv.org/abs/1707.07045) - foundation for modern neural coreference.
- [SpanBERT](https://arxiv.org/abs/1907.10529) - span-based pretraining; strong coreference baseline.
- [coref-hoi](https://github.com/lxucs/coref-hoi) - higher-order inference coreference.
- [maverick-coref](https://github.com/SapienzaNLP/maverick-coref) (2024) - efficient coreference matching the best larger systems.
- [LingMess](https://arxiv.org/abs/2205.12644) - linguistically-motivated category-based coreference scoring.
LLM-based:

- [LLMs for Coreference Resolution](https://arxiv.org/abs/2310.05884) - prompting and fine-tuning for coreference.
- [Multilingual Coreference Shared Task: Can LLMs Dethrone Traditional Approaches?](https://arxiv.org/abs/2509.17796) (2025) - 9 systems across 4 LLM-based and 5 traditional approaches; traditional methods still lead but LLMs are closing the gap.

### Text Classification and Sentiment Analysis

[Back to Top](#contents)

- [fastText classifier](https://arxiv.org/abs/1607.01759) - strong, fast linear baseline.
- [Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/) - canonical fine-grained sentiment dataset.
- [SetFit](https://github.com/huggingface/setfit) - few-shot text classification without prompts.
- [FastFit](https://github.com/IBM/fastfit) - fast few-shot for many-class settings.
- [SST / IMDB / AG News with DeBERTa-v3](https://arxiv.org/abs/2111.09543) - current encoder-fine-tuning baseline.
- [PySS3](https://github.com/sergioburdisso/pyss3) - white-box, interpretable text classifier.
- [LLMs as Annotators](https://arxiv.org/abs/2305.13734) - using LLMs for text classification labeling, with caveats.

### Topic Modeling

[Back to Top](#contents)

- [Latent Dirichlet Allocation (Blei et al.)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) - foundational topic model.
- [gensim](https://radimrehurek.com/gensim/) - LDA, LSI, HDP in Python.
- [BigARTM](https://github.com/bigartm/bigartm) - fast regularized topic modeling.
- [BERTopic](https://github.com/MaartenGr/BERTopic) - clustering-based topic modeling on top of contextual embeddings; common modern default.
- [Top2Vec](https://github.com/ddangelov/Top2Vec) - jointly learns topic and document vectors.
- [CorEx Topic](https://github.com/gregversteeg/corex_topic) - hierarchical topic modeling with anchor words.

### Summarization

[Back to Top](#contents)

- [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) - extractive graph-based summarization.
- [Pointer-Generator Networks (See et al.)](https://arxiv.org/abs/1704.04368) - foundational neural abstractive summarization.
- [PEGASUS](https://arxiv.org/abs/1912.08777) - gap-sentences pretraining for summarization.
- [BART](https://arxiv.org/abs/1910.13461) - widely used denoising seq2seq baseline.
- [BookSum](https://arxiv.org/abs/2105.08209) and [SCROLLS](https://arxiv.org/abs/2201.03533) - long-document summarization benchmarks.
LLM-based:

- [Benchmarking LLMs for News Summarization](https://arxiv.org/abs/2301.13848) - LLMs vs fine-tuned summarizers.
- [Element-Aware Summarization with LLMs](https://arxiv.org/abs/2305.13412) - structured prompting for summarization.
- [Understanding LLM Reasoning for Abstractive Summarization](https://arxiv.org/abs/2512.03503) (2025) - explicit reasoning improves fluency but hurts factual grounding; longer reasoning budgets can harm faithfulness.

### Machine Translation

[Back to Top](#contents)

Statistical and foundational neural:

- [Moses](http://statmt.org/moses/) - reference statistical MT system.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - transformer; reset the field.
- [Marian NMT](https://github.com/marian-nmt/marian) - efficient C++ NMT framework.
- [Fairseq](https://github.com/facebookresearch/fairseq) - PyTorch sequence modeling toolkit.

Massively multilingual:

- [NLLB-200](https://arxiv.org/abs/2207.04672) - MT for 200 languages.
- [MADLAD-400](https://arxiv.org/abs/2309.04662) - 400+ language MT.
- [SeamlessM4T](https://arxiv.org/abs/2312.05187) - speech and text MT, 100+ languages.

Evaluation:

- [COMET](https://github.com/Unbabel/COMET) - learned MT metric; current de-facto standard alongside chrF.
- [sacrebleu](https://github.com/mjpost/sacrebleu) - reproducible BLEU/chrF/TER scoring.
- [BERTScore](https://github.com/Tiiiger/bert_score) - similarity-based generation metric.

LLM-based:

- [Is ChatGPT a Good Translator?](https://arxiv.org/abs/2301.08745) - LLMs as machine translation systems.
- [Adapting LLMs for Document-Level MT](https://arxiv.org/abs/2401.06468) (2024) - LLMs for context-aware translation.
- [GPT-4 vs Human Translators](https://arxiv.org/abs/2308.03245) - quality comparison on professional MT.
- [Multilingual MT with Open LLMs at Practical Scale](https://arxiv.org/abs/2502.02481) (2025) - benchmarks sub-10B open LLMs on 28-language MT; matches GPT-4-turbo and Google Translate.
- [Bridging the Linguistic Divide: Survey on LLMs for MT](https://arxiv.org/abs/2504.01919) (2025) - survey of how instruction-following, in-context learning, and preference alignment have restructured MT methodology.

### Question Answering and Reading Comprehension

[Back to Top](#contents)

Datasets and foundational systems:

- [SQuAD / SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) - extractive reading comprehension.
- [Natural Questions](https://ai.google.com/research/NaturalQuestions/) - real-user questions over Wikipedia.
- [HotpotQA](https://hotpotqa.github.io/) - multi-hop reasoning.
- [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) - distantly-supervised QA.
- [DrQA](https://github.com/facebookresearch/DrQA) - open-domain QA over Wikipedia.
- [Document-QA](https://github.com/allenai/document-qa) - multi-paragraph reading comprehension.

Modern open-domain QA:

- [DPR](https://arxiv.org/abs/2004.04906) and [FiD](https://arxiv.org/abs/2007.01282) - retrieve-then-read; the standard pre-LLM open-domain QA pipeline.
- [Atlas](https://arxiv.org/abs/2208.03299) - retrieval-augmented LM for few-shot QA.
- See also [Retrieval for NLP](#retrieval-for-nlp).

LLM-era:

- [GPT-4 with retrieval on TriviaQA / NQ](https://arxiv.org/abs/2305.06983)
- [Self-RAG](https://arxiv.org/abs/2310.11511) (2023) - retrieval, generation, and self-critique.
- [GAIA](https://arxiv.org/abs/2311.12983) - general AI assistant benchmark including multi-step QA.

### Information Extraction Beyond NER

[Back to Top](#contents)

- [OpenIE 6](https://github.com/dair-iitd/openie6) - schema-free open information extraction.
- [Template-Based Information Extraction without the Templates](https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
- [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
- [REBEL](https://github.com/Babelscape/rebel) - end-to-end relation extraction.
- [DocRED](https://github.com/thunlp/DocRED) - document-level relation extraction benchmark.
- [LLMs for Semantic Role Labeling](https://arxiv.org/abs/2506.05385) (2025) - generative LLMs with RAG and self-correction surpass encoder-decoder BERT-style models on SRL in English and Chinese.
- [Adapting LLMs for Minimal-edit GEC](https://arxiv.org/abs/2506.13148) (2025) - decoder-only LLMs with a novel error-rate adaptation schedule set new SOTA on BEA-test grammatical error correction.

### Retrieval and Embeddings

[Back to Top](#contents)

Dense and late-interaction retrieval, increasingly the substrate for QA and IR:

- [DPR (Dense Passage Retrieval)](https://arxiv.org/abs/2004.04906) - dual-encoder retrieval baseline.
- [ColBERT](https://arxiv.org/abs/2004.12832) and [ColBERTv2](https://arxiv.org/abs/2112.01488) - late-interaction retrieval; strong on out-of-domain.
- [E5](https://arxiv.org/abs/2212.03533) and [E5-Mistral](https://arxiv.org/abs/2401.00368) - widely used dense embedding families.
- [BGE](https://github.com/FlagOpen/FlagEmbedding) and [BGE-M3](https://arxiv.org/abs/2402.03216) (2024) - multilingual, multi-functionality embeddings; top of MTEB across languages.
- [Nomic Embed](https://arxiv.org/abs/2402.01613) (2024) - fully open, reproducible embedding model.
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - nested embeddings supporting variable dimensionality at inference.
- [GritLM](https://arxiv.org/abs/2402.09906) (2024) - unified generation and embedding from one model.
- [RAG (Retrieval-Augmented Generation)](https://arxiv.org/abs/2005.11401) - the original retrieval-augmented framework; foundation for modern QA pipelines.
- [Gemini Embedding](https://arxiv.org/abs/2503.07891) (2025) - Gemini-derived dense embeddings; SOTA on MMTEB across 250+ languages and on cross-lingual retrieval (XOR-Retrieve, XTREME-UP).
- [Qwen3-Embedding](https://arxiv.org/abs/2506.05176) (2025) - decoder-based embedding series (0.6B-8B) built on Qwen3; #1 on MTEB Multilingual and MTEB Code, surpassing prior proprietary models.
- [Rank1](https://arxiv.org/abs/2502.18418) (2025) - first reranking model trained with test-time compute via DeepSeek-R1 reasoning-trace distillation; SOTA on instruction-following and OOD retrieval.
- [ReasonEmbed](https://arxiv.org/abs/2510.08252) (2025) - embedding model for reasoning-intensive retrieval with ReMixer data synthesis and Redapter adaptive training; record nDCG@10 of 38.1 on BRIGHT.
- [ColBERT-Att](https://arxiv.org/abs/2603.25248) (2026) - extends late-interaction retrieval by integrating query and document attention weights into ColBERT scoring; improves recall on MS-MARCO, BEIR, and LoTTE.
Embedding and retrieval benchmarks:

- [MMTEB](https://arxiv.org/abs/2502.13595) (2025) - community expansion of MTEB to 500+ tasks across 250+ languages.

### Speech and Text

[Back to Top](#contents)

A short pointer set, since this borders adjacent fields:

- [Whisper](https://github.com/openai/whisper) - multilingual ASR; the modern open default.
- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) - unified speech and text translation.
- [Canary](https://huggingface.co/nvidia/canary-1b) (NVIDIA, 2024) - top open multilingual ASR model.
- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) - foundational self-supervised speech pretraining.
- [Coqui TTS](https://github.com/coqui-ai/TTS) and [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) - open TTS.

## Datasets

[Back to Top](#contents)

Dataset hubs and lists:

- [HuggingFace Datasets Hub](https://huggingface.co/datasets) - the central index for modern NLP datasets, with versioned, streamable loaders.
- [nlp-datasets](https://github.com/niderhoff/nlp-datasets) - large collection of NLP datasets.
- [gensim-data](https://github.com/RaRe-Technologies/gensim-data) - data repository for pretrained NLP models and NLP corpora.

Pretraining-scale corpora (open):

- [The Pile](https://pile.eleuther.ai/) - 825 GiB diverse text corpus.
- [RedPajama / RedPajama-V2](https://github.com/togethercomputer/RedPajama-Data) (2023-2024) - reproductions of LLaMA pretraining data; V2 is 30T tokens with quality signals.
- [Dolma](https://github.com/allenai/dolma) (AI2, 2023-2024) - 3T-token open pretraining corpus with documented filtering pipeline.
- [FineWeb / FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (2024) - 15T-token cleaned web corpus; FineWeb-Edu filters for educational quality.
- [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) - 6.3T tokens across 167 languages.
- [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) (2024) - 2T-token open-license multilingual corpus.

Task and instruction datasets:

- [Universal Dependencies](https://universaldependencies.org/) - cross-linguistically consistent treebank annotation, 100+ languages.
- [Tülu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) (2024) - open instruction-tuning data behind Tülu 3.
- [tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp/) - tiny NLP multi-lingual QA datasets and library to generate your own synthetic copies.

## Multilingual NLP Frameworks

[Back to Top](#contents)

- [UDPipe](https://github.com/ufal/udpipe) is a trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U files. Primarily written in C++, offers a fast and reliable solution for multilingual NLP processing.
- [NLP-Cube](https://github.com/adobe/NLP-Cube) : Natural Language Processing Pipeline - Sentence Splitting, Tokenization, Lemmatization, Part-of-speech Tagging and Dependency Parsing. New platform, written in Python with Dynet 2.0. Offers standalone (CLI/Python bindings) and server functionality (REST API).
- [UralicNLP](https://github.com/mikahama/uralicNLP) is an NLP library mostly for many endangered Uralic languages such as Sami languages, Mordvin languages, Mari languages, Komi languages and so on. Also some non-endangered languages are supported such as Finnish together with non-Uralic languages such as Swedish and Arabic. UralicNLP can do morphological analysis, generation, lemmatization and disambiguation.

## Language Models for NLP

[Back to Top](#contents)

Pretrained language models and the research around them, scoped to NLP tasks and linguistic phenomena. For general-purpose LLM tooling, agents, or RAG application kits, see [See Also](#see-also).

### Pretraining and Adaptation

Encoders (still the workhorse for classical NLP tasks):

- [BERT](https://arxiv.org/abs/1810.04805) - bidirectional transformer pretraining; foundation for most encoder-based NLP work since 2018.
- [RoBERTa](https://arxiv.org/abs/1907.11692) - robustly optimized BERT pretraining; common encoder baseline.
- [DeBERTa / DeBERTa-v3](https://arxiv.org/abs/2111.09543) - disentangled attention; strong on classification, NER, NLI.
- [ELECTRA](https://arxiv.org/abs/2003.10555) - replaced-token-detection pretraining, sample-efficient.
- [ModernBERT](https://arxiv.org/abs/2412.13663) (2024) - modernized encoder with rotary embeddings, FlashAttention, 8K context; current go-to encoder for classification, NER, retrieval.
- [NeoBERT](https://arxiv.org/abs/2502.19587) (2025) - 250M-parameter encoder integrating modern architecture improvements (RoPE, 4K context, optimized depth-to-width); state of the art on MTEB, surpasses ModernBERT and RoBERTa-large under identical fine-tuning.

Encoder-decoder and seq2seq:

- [T5](https://arxiv.org/abs/1910.10683) and [FLAN-T5](https://arxiv.org/abs/2210.11416) - text-to-text framing for NLP tasks; strong instruction-tuned encoder-decoder baselines.
- [BART](https://arxiv.org/abs/1910.13461) - denoising seq2seq pretraining; widely used for summarization and generation.

Open decoder-only LMs (used as substrate for NLP tasks):

- [Llama 3 / 3.1 / 3.3](https://arxiv.org/abs/2407.21783) (Meta, 2024-2025) - widely adopted open-weight family; default base for fine-tuning across NLP tasks.
- [Qwen 2.5 / Qwen 3](https://qwenlm.github.io/) (Alibaba, 2024-2025) - strong multilingual coverage, especially Chinese; often top open model on multilingual benchmarks.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2024) - efficient MoE pretraining; competitive open base model.
- [OLMo 2](https://arxiv.org/abs/2501.00656) (AI2, 2025) - fully open: weights, training data, code; reproducibility benchmark.
- [Gemma 2 / Gemma 3](https://arxiv.org/abs/2408.00118) (Google, 2024-2025) - open small/mid-size models with strong NLP-task performance.
- [Mistral / Mixtral](https://arxiv.org/abs/2401.04088) - efficient dense and sparse-MoE open models.
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832) - encoder vs decoder vs encoder-decoder for NLP transfer.

### Multilingual and Cross-Lingual Models

- [XLM-R](https://arxiv.org/abs/1911.02116) - cross-lingual masked LM trained on CommonCrawl, 100 languages.
- [mT5](https://arxiv.org/abs/2010.11934) - multilingual T5 covering 101 languages.
- [BLOOM](https://arxiv.org/abs/2211.05100) - 176B-parameter open multilingual LM, 46 natural languages.
- [Aya 23 / Aya Expanse](https://arxiv.org/abs/2412.04261) (Cohere For AI, 2024) - massively multilingual instruction-tuned models covering 23-101 languages.
- [Glot500](https://arxiv.org/abs/2305.12182) - encoder for 500+ languages, focus on low-resource.
- [NLLB-200](https://arxiv.org/abs/2207.04672) - No Language Left Behind: MT for 200 languages.
- [MADLAD-400](https://arxiv.org/abs/2309.04662) - 400+ language MT model and 3T-token multilingual corpus.
- [SeamlessM4T / Seamless](https://arxiv.org/abs/2312.05187) (Meta, 2023-2024) - multilingual and multimodal speech-text translation, 100+ languages.
- [SEA-LION / SeaLLM](https://arxiv.org/abs/2312.00738) (2024-2025) - LMs targeting Southeast Asian languages.
- [Babel](https://arxiv.org/abs/2503.00865) (2025) - open multilingual LLMs (9B and 83B) covering the top 25 languages by speaker population (~90% of global speakers); surpasses comparably-sized open multilingual models on XCOPA, XNLI, MGSM, FLORES-200.
- [Lugha-Llama](https://arxiv.org/abs/2504.06536) (Princeton/Mila, 2025) - Llama-3.1-8B adapted for low-resource African languages via the curated WURA corpus; SOTA open-source results on IrokoBench and AfriQA.
- [AfriqueLLM](https://arxiv.org/abs/2601.06395) (McGill, 2026) - suite of open LLMs (4B-14B) continued-pretrained on 26B tokens across 20 African languages with a comprehensive empirical study of data mixing.
- [TranslateGemma](https://arxiv.org/abs/2601.09012) (Google, 2026) - open translation-specialized models built on Gemma 3, covering 55 language pairs via SFT and RL with quality-reward models.
- [MiLMMT-46](https://arxiv.org/abs/2602.11961) (Xiaomi, 2026) - open multilingual MT scaled across 46 languages, matching commercial systems like Google Translate and Gemini 3 Pro.

### Evaluation and Benchmarks

NLU and cross-lingual:

- [GLUE](https://gluebenchmark.com/) and [SuperGLUE](https://super.gluebenchmark.com/) - English NLU benchmarks.
- [XTREME](https://sites.research.google/xtreme) and [XGLUE](https://microsoft.github.io/XGLUE/) - cross-lingual NLU.
- [XNLI](https://github.com/facebookresearch/XNLI) - cross-lingual natural language inference, 15 languages.
- [FLORES-200](https://github.com/facebookresearch/flores) - MT evaluation across 200 languages.
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Massive Text Embedding Benchmark; standard for sentence/document encoders.
- [BEIR](https://github.com/beir-cellar/beir) - heterogeneous IR benchmark for retrieval models.

Modern LM evaluation (2023-2026):

- [HELM](https://crfm.stanford.edu/helm/) - holistic evaluation across NLP tasks, accuracy and beyond.
- [BIG-bench](https://github.com/google/BIG-bench) - 200+ tasks probing language model capabilities.
- [MMLU](https://github.com/hendrycks/test) - multitask knowledge evaluation across 57 subjects.
- [MMLU-Pro](https://arxiv.org/abs/2406.01574) (2024) - harder, more discriminative successor to MMLU.
- [GPQA](https://arxiv.org/abs/2311.12022) - graduate-level Q&A, "Google-proof" reasoning evaluation.
- [IFEval](https://arxiv.org/abs/2311.07911) - verifiable instruction-following evaluation.
- [Chatbot Arena (LMSYS)](https://lmarena.ai/) - human-preference ELO leaderboard for chat models.
- [LiveBench](https://livebench.ai/) (2024) - contamination-resistant benchmark with monthly refresh.
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - unified framework for LM benchmark evaluation.
- [MMLU-ProX](https://arxiv.org/abs/2503.10497) (2025) - multilingual extension of MMLU-Pro to 29 typologically diverse languages; reveals up to 24.3% performance gap between high- and low-resource languages.
- [MultiChallenge](https://arxiv.org/abs/2501.17399) (2025) - multi-turn conversational benchmark exposing simultaneous instruction-following and in-context-reasoning failures; all tested frontier models score below 50%.
- [FRAMES](https://arxiv.org/abs/2409.12941) (2025) - unified RAG evaluation: 824 multi-hop questions requiring factuality, retrieval accuracy, and cross-document reasoning together.

Long-context evaluation:

- [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) - retrieval probe for long-context windows.
- [RULER](https://arxiv.org/abs/2404.06654) (2024) - synthetic long-context tasks beyond simple retrieval.
- [LongBench](https://github.com/THUDM/LongBench) - bilingual long-context benchmark across NLP tasks.
- [LongBench v2](https://arxiv.org/abs/2412.15204) (2025) - 503 expert-crafted multiple-choice questions spanning 8K-2M-word contexts with deep multi-hop reasoning; humans score 53.7% under time pressure.
- [U-NIAH](https://arxiv.org/abs/2503.00353) (2025) - extends needle-in-haystack with multi-needle and nested configurations; shows RAG mitigates lost-in-the-middle for smaller LLMs but degrades reasoning models.

### Reasoning and Test-Time Compute

A trend-defining direction in 2024-2026: models that produce explicit reasoning traces and benefit from extra inference compute.

- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - foundational result; intermediate reasoning steps improve performance.
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - majority vote over sampled CoT chains.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) - search over reasoning trees.
- [Self-Refine](https://arxiv.org/abs/2303.17651) and [Reflexion](https://arxiv.org/abs/2303.11366) - self-correction at inference time.
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - chain-of-thought for NLP reasoning tasks.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - process-supervised reward models for reasoning.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025) - open reasoning model trained with pure RL; replicated o1-style behavior in the open.
- [OpenAI o1 / o3](https://openai.com/index/learning-to-reason-with-llms/) (2024-2025) - test-time-compute reasoning systems.
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) (2024) - systematic study of inference-time compute tradeoffs.
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) (2025) - small open reasoning recipe via budget-forcing.
- [Kimi k1.5](https://arxiv.org/abs/2501.12599) (2025) - long-context RL with policy optimization (no MCTS, no PRM) reaching o1-level performance; introduces long-CoT distillation into short-CoT models.
- [rStar-Math](https://arxiv.org/abs/2501.04519) (2025) - small policy model paired with a process preference model trained via MCTS rollouts; enables small LMs to bootstrap reasoning without distilling from larger models.
- [DAPO](https://arxiv.org/abs/2503.14476) (2025) - open GRPO-based RL training system with four key improvements (decoupled clipping, dynamic sampling, token-level loss, entropy bonus); reproduces and surpasses DeepSeek-R1-Zero-level reasoning.
- [VAPO](https://arxiv.org/abs/2504.05118) (2025) - value-model-based RL with length-adaptive GAE and token-level clipping; surpasses value-free GRPO methods on AIME 2024 with stable training.
- [ThinkPRM](https://arxiv.org/abs/2504.16828) (2025) - generative process reward models that produce chain-of-thought verification per step, matching discriminative PRMs with 1% of the supervision labels.
- [OpenThoughts](https://arxiv.org/abs/2506.04178) (2025) - 1000+ controlled experiments on data recipes for open reasoning models; SOTA on AIME 2025 matching closed distillation baselines.

### Long Context and Alternative Architectures

- [Mamba](https://arxiv.org/abs/2312.00752) and [Mamba-2](https://arxiv.org/abs/2405.21060) - selective state-space models, linear-time long-context alternative to attention.
- [RWKV](https://arxiv.org/abs/2305.13048) - RNN-transformer hybrid scaling to large parameter counts.
- [Jamba](https://arxiv.org/abs/2403.19887) (2024) - hybrid Mamba-Transformer-MoE architecture.
- [RoPE](https://arxiv.org/abs/2104.09864) and [YaRN](https://arxiv.org/abs/2309.00071) - rotary position embeddings and context-length extension.
- [Position Interpolation](https://arxiv.org/abs/2306.15595) - extending context windows with minimal fine-tuning.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) - long-context degradation patterns in NLP tasks.
- [RAG vs Long-Context LLMs](https://arxiv.org/abs/2407.16833) (2024) - tradeoffs for QA over long inputs.
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (2025) - neural long-term memory module that learns to memorize historical context at test time; scales beyond 2M tokens, outperforms transformers and modern linear-recurrent models on language modeling and reasoning.
- [MiniMax-01](https://arxiv.org/abs/2501.08313) (2025) - 456B-parameter hybrid combining lightning (linear) attention with sparse softmax attention; matches GPT-4o-level NLP performance at up to 4M-token inference contexts.
- [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) (2025) - trainable sparse attention combining coarse-grained compression with fine-grained selection; large speedups at 64K with no NLP-benchmark degradation.
- [LongRoPE2](https://arxiv.org/abs/2502.20082) (2025) - identifies undertraining of high-frequency RoPE dimensions and applies evolutionary-search rescaling; extends LLaMA3-8B to 128K with 80x fewer training tokens than Meta's recipe.
- [Characterizing SSM and Hybrid LM Long-Context Performance](https://arxiv.org/abs/2507.12442) (2025) - first comprehensive memory and speed analysis of transformer, SSM, and hybrid models up to 220K tokens; SSMs are up to 4x faster, hybrids balance recall and efficiency.

### Factuality, Hallucination, Calibration

- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) - taxonomy and mitigation strategies.
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) - benchmark for truthfulness in question answering.
- [FActScore](https://github.com/shmsw25/FActScore) - fine-grained factual precision in long-form generation.
- [LongFact / SAFE](https://arxiv.org/abs/2403.18802) (2024) - long-form factuality benchmark and search-augmented evaluator.
- [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt) - sampling-based hallucination detection.
- [RAGAS](https://github.com/explodinggradients/ragas) - reference-free evaluation for RAG and QA pipelines.
- [Lookback Lens](https://arxiv.org/abs/2407.07071) (2024) - attention-pattern-based hallucination detection in long-context generation.
- [Calibration of LLMs on Multiple Choice](https://arxiv.org/abs/2402.13887) (2024) - calibration analysis under format effects.
- [HalluLens](https://arxiv.org/abs/2504.17550) (2025) - hallucination benchmark with extrinsic/intrinsic taxonomy and dynamic test-set regeneration to resist data leakage.
- [Atomic Calibration](https://arxiv.org/abs/2410.13246) (2025) - claim-level calibration analysis for long-form generation; models are substantially worse-calibrated on extended outputs than on single claims.
- [FRANQ](https://arxiv.org/abs/2505.21072) (2025) - faithfulness-aware uncertainty quantification for RAG fact-checking; formally separates faithfulness from factuality.
- [MUCH](https://arxiv.org/abs/2511.17081) (2025) - multilingual claim-hallucination benchmark across English, French, Spanish, German with token-level logits released for principled UQ evaluation.
- [HalluHard](https://arxiv.org/abs/2602.01031) (2026) - hard multi-turn hallucination benchmark for citation-required responses; ~30% hallucination rates persist even with web search.
- [CURE: Think Through Uncertainty](https://arxiv.org/abs/2604.12046) (2026) - trains models to reason about claim-level uncertainty before generating; large gains on biography factuality and FactBench AUROC.

### Probing and Interpretability

- [A Primer in BERTology](https://arxiv.org/abs/2002.12327) - what BERT learns about language.
- [Probing Classifiers (Belinkov)](https://arxiv.org/abs/2102.12452) - methodology, limitations, alternatives.
- [Locating and Editing Factual Associations in GPT (ROME)](https://rome.baulab.info/) - causal tracing of factual recall.
- [The Pyramid of NLP Probes](https://arxiv.org/abs/2104.07885) - structural probing for linguistic knowledge.
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/) - foundation for the sparse-feature view of transformer representations.
- [Towards Monosemanticity / Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Anthropic, 2024) - sparse autoencoders extracting interpretable features from production-scale LMs.
- [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) - SAE methodology for LM interpretability.
- [Neuronpedia](https://www.neuronpedia.org/) - open platform browsing SAE features across models.
- [Influence Functions Scale to LLMs](https://arxiv.org/abs/2308.03296) (2023) - identifying training examples driving model behavior.
- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Anthropic, 2025) - introduces cross-layer transcoders and attribution graphs to construct an interpretable replacement model; enables prompt-level circuit tracing of feature-to-feature causal interactions.
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic, 2025) - applies attribution graphs to Claude 3.5 Haiku across multi-hop reasoning, rhyme planning, and jailbreak case studies.
- [Transcoders Beat Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2501.18823) (2025) - shows transcoders (reconstructing layer outputs from inputs) yield more interpretable features than SAEs; introduces skip transcoders.
- [Survey on Sparse Autoencoders for LLM Interpretability](https://arxiv.org/abs/2503.05613) (EMNLP 2025) - reference survey of SAE architectures, training strategies, feature explanation, and evaluation.
- [Finding Highly Interpretable Prompt-Specific Circuits](https://arxiv.org/abs/2602.13483) (2026) - identifies circuits at the per-prompt level (rather than per-task); reveals mechanism clustering by prompt family.

### Efficient and Small Language Models

Distillation and small models:

- [DistilBERT](https://arxiv.org/abs/1910.01108) and [MiniLM](https://arxiv.org/abs/2002.10957) - distilled encoders for production NLP.
- [Phi-3 / Phi-4](https://arxiv.org/abs/2412.08905) (Microsoft, 2024) - small models trained on curated data, competitive with much larger ones on NLP benchmarks.
- [SmolLM2](https://arxiv.org/abs/2502.02737) (HuggingFace, 2025) - fully open small-LM family with reproducible training data.
- [SmolLM3](https://huggingface.co/blog/smollm3) (HuggingFace, 2025) - 3B fully open decoder pretrained on 11.2T tokens with NoPE and YaRN for 128K context; competitive with 4B-class models.
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) (Google, 2025) - 1B-27B open models with high local-to-global attention ratio to keep KV-cache tractable at 128K context.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Alibaba, 2025) - dense and MoE models 0.6B-235B with unified thinking/non-thinking modes; the 30B-A3B MoE matches larger dense models while activating only 3B parameters.
- [Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2507.13575) (Apple, 2025) - on-device 3B model using KV-cache sharing and 2-bit QAT for 37.5% cache memory reduction without accuracy loss.
- [Sentence-Transformers](https://www.sbert.net/) - sentence and paragraph embeddings via Siamese BERT.
- [SetFit](https://github.com/huggingface/setfit) - few-shot text classification without prompts.
- [FastFit](https://github.com/IBM/fastfit) - fast few-shot classification for many-class settings.
- [GTE](https://huggingface.co/thenlper/gte-base), [BGE](https://github.com/FlagOpen/FlagEmbedding), and [Stella](https://huggingface.co/dunzhang/stella_en_1.5B_v5) - compact text embedding models near the top of MTEB.

Quantization and serving (relevant when deploying NLP models at scale):

- [GPTQ](https://arxiv.org/abs/2210.17323) - post-training quantization for transformers.
- [AWQ](https://arxiv.org/abs/2306.00978) - activation-aware weight quantization.
- [KVTuner](https://arxiv.org/abs/2502.04420) (ICML 2025) - sensitivity-aware layer-wise mixed-precision KV-cache quantization; up to 21% throughput improvement over uniform KV8.
- [GGUF / llama.cpp](https://github.com/ggerganov/llama.cpp) - portable quantized inference.
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention-based high-throughput LM serving.
- [SGLang](https://github.com/sgl-project/sglang) - structured generation and efficient serving.
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) - HF production serving for LMs.

Parameter-efficient fine-tuning:

- [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) - low-rank adapters and quantized fine-tuning; the standard for adapting LMs to NLP tasks on modest hardware.
- [DoRA](https://arxiv.org/abs/2402.09353) (2024) - weight-decomposed low-rank adaptation.
- [PEFT](https://github.com/huggingface/peft) - HuggingFace library bundling LoRA, prefix tuning, IA3, and others.

### Instruction Tuning and Preference Optimization

- [FLAN](https://arxiv.org/abs/2109.01652) - finetuned language models as zero-shot learners.
- [InstructGPT](https://arxiv.org/abs/2203.02155) - training LMs to follow instructions with human feedback.
- [Self-Instruct](https://github.com/yizhongw/self-instruct) - bootstrapping instruction data from LMs.
- [Super-NaturalInstructions](https://github.com/allenai/natural-instructions) - 1600+ NLP tasks with instructions.
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - training LMs with AI-generated feedback against a written constitution.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - simpler alternative to RLHF; widely adopted.
- [Tülu 3](https://arxiv.org/abs/2411.15124) (AI2, 2024) - fully open post-training recipe with state-of-the-art results among open models.
- [LIMA](https://arxiv.org/abs/2305.11206) - "less is more for alignment"; small high-quality SFT data goes a long way.
- [TRL](https://github.com/huggingface/trl) - reference library for SFT, DPO, GRPO, and RLHF.
- [Magpie](https://arxiv.org/abs/2406.08464) (2024-2025) - synthesizes high-quality instruction-response pairs by prompting aligned LMs with nothing; SFT on the filtered subset matches official Llama-3-Instruct.

### Bias, Fairness, Safety in NLP

- [StereoSet](https://github.com/moinnadeem/StereoSet) - measuring stereotypical bias in pretrained LMs.
- [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) - social bias measurement in masked LMs.
- [WinoBias](https://github.com/uclanlp/corefBias) - gender bias in coreference resolution.
- [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP) - bias measurement across many demographic axes.
- [RealToxicityPrompts](https://github.com/allenai/real-toxicity-prompts) - toxicity in LM generation.
- [Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) - models tailoring answers to user beliefs.
- [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093) (Anthropic, 2024) - models strategically complying during training.
- [WildGuard](https://arxiv.org/abs/2406.18495) (2024) - open safety moderation model and benchmark.
- [Emergent Misalignment](https://arxiv.org/abs/2502.17424) (2025) - finetuning on a narrow task (insecure code) unexpectedly produces broad alignment failures across unrelated domains.
- [SafeDialBench](https://arxiv.org/abs/2502.11090) (2025) - multilingual (Chinese/English) safety benchmark of 4000+ multi-turn dialogues across 22 scenarios and 7 jailbreak strategies.
- [TeleAI-Safety](https://arxiv.org/abs/2512.05485) (2025) - modular jailbreak evaluation framework integrating 19 attacks, 29 defenses, and 19 evaluation methods across 14 models and 12 risk categories.
- [IndicSafe](https://arxiv.org/abs/2603.17915) (2026) - multilingual safety benchmark across 12 Indic languages; reveals 12.8% cross-language agreement, with over-refusal in low-resource scripts.
- [VLAF: Value-Conflict Alignment Faking](https://arxiv.org/abs/2604.20995) (2026) - alignment faking occurs in models as small as 7B in 37% of cases when policy conflicts with internalized values; steering-vector mitigation reduces it 94%.

## NLP per Language

[Back to Top](#contents)

Resources organized by human language. Click a section to expand.

<details>
<summary>

### NLP in Arabic

</summary>

[Back to Top](#contents)

### Libraries

- [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) - Python toolkit for Arabic NLP including dialect ID, morphology, NER.
- [goarabic](https://github.com/01walid/goarabic) - Go package for Arabic text processing.
- [jsastem](https://github.com/ejtaal/jsastem) - JavaScript Arabic stemmer.
- [PyArabic](https://pypi.org/project/PyArabic/) - Python library for Arabic.
- [RFTokenizer](https://github.com/amir-zeldes/RFTokenizer) - trainable segmenter for Arabic, Hebrew, and Coptic.
- [Farasa](https://farasa.qcri.org/) - QCRI segmentation, POS tagging, and NER for Arabic.

### Models and Embeddings

- [AraBERT](https://github.com/aub-mind/arabert) - Arabic BERT family.
- [CAMeLBERT](https://github.com/CAMeL-Lab/CAMeLBERT) - BERT models for MSA, dialectal, and Classical Arabic.
- [AraELECTRA](https://aclanthology.org/2021.wanlp-1.20/) - efficient Arabic pretraining (released alongside [AraBERT](https://github.com/aub-mind/arabert)).
- [Jais](https://huggingface.co/inceptionai/jais-13b) (2023-2024) - bilingual Arabic-English open LM family.
- [ALLaM](https://arxiv.org/abs/2407.15390) (SDAIA, 2024) - Arabic-first foundation models.

### Datasets

- [Multidomain Datasets](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces) - largest available multi-domain Arabic sentiment analysis resources.
- [LABR](https://github.com/mohamedadaly/labr) - large Arabic book reviews dataset.
- [Arabic Stopwords](https://github.com/mohataher/arabic-stop-words) - aggregated Arabic stopwords.
- [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) (2024) - Arabic MMLU benchmark.

</details>

<details>
<summary>

### NLP in Chinese

</summary>

[Back to Top](#contents)

### Libraries

- [jieba](https://github.com/fxsjy/jieba#jieba-1) - Python package for Chinese word segmentation.
- [SnowNLP](https://github.com/isnowfy/snownlp) - Python package for Chinese NLP.
- [FudanNLP](https://github.com/FudanNLP/fnlp) - Java library for Chinese text processing.
- [HanLP](https://github.com/hankcs/HanLP) - multilingual NLP library with strong Chinese support.
- [LTP](https://github.com/HIT-SCIR/ltp) - HIT Language Technology Platform: segmentation, POS, NER, parsing.

### Models and Embeddings

- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) - whole-word masking BERT for Chinese.
- [MacBERT](https://github.com/ymcui/MacBERT) - improved Chinese BERT with MLM-as-correction pretraining.
- [Qwen 2.5 / Qwen 3](https://github.com/QwenLM/Qwen3) - Alibaba's open Chinese-strong LM family.
- [ChatGLM3 / GLM-4](https://github.com/THUDM/ChatGLM3) - Tsinghua's bilingual Chinese-English LMs.
- [Baichuan 2](https://github.com/baichuan-inc/Baichuan2) - open Chinese LM.
- [Yi](https://github.com/01-ai/Yi) - 01.AI's bilingual open LMs.
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) - efficient open MoE model with strong Chinese.

### Anthology

- [funNLP](https://github.com/fighting41love/funNLP) - large collection of Chinese NLP tools and resources.

</details>

<details>
<summary>

### NLP in Danish

</summary>

[Back to Top](#contents)

- [Named Entity Recognition for Danish](https://github.com/ITUnlp/daner)
- [DaNLP](https://github.com/alexandrainst/danlp) - NLP resources in Danish.
- [Awesome Danish](https://github.com/fnielsen/awesome-danish) - curated list of resources for Danish language technology.

</details>

<details>
<summary>

### NLP in Dutch

</summary>

[Back to Top](#contents)

- [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch (POS tagging, lemmatization, dependency parsing, NER).
- [SimpleNLG_NL](https://github.com/rfdj/SimpleNLG-NL) - Dutch surface realiser for natural language generation, based on the SimpleNLG implementation.
- [Alpino](https://github.com/rug-compling/alpino) - dependency parser for Dutch (also does POS tagging and lemmatization).
- [Kaldi NL](https://github.com/opensource-spraakherkenning-nl/Kaldi_NL) - Dutch speech-recognition models based on [Kaldi](http://kaldi-asr.org/).
- [spaCy Dutch model](https://spacy.io/models/nl) - industrial-strength NLP with a Dutch pipeline.

</details>

<details>
<summary>

### NLP in German

</summary>

[Back to Top](#contents)

- [German-NLP](https://github.com/adbar/German-NLP) - curated list of open-access, open-source, and off-the-shelf resources and tools developed with a focus on German.

</details>

<details>
<summary>

### NLP in Hungarian

</summary>

[Back to Top](#contents)

- [awesome-hungarian-nlp](https://github.com/oroszgy/awesome-hungarian-nlp) - curated list of free resources for Hungarian NLP.

</details>

<details>
<summary>

### NLP in Indic Languages

</summary>

[Back to Top](#contents)

### Data, Corpora and Treebanks

- [Hindi Dependency Treebank](https://ltrc.iiit.ac.in/treebank_H2014/) - A multi-representational multi-layered treebank for Hindi and Urdu
- [Universal Dependencies Treebank in Hindi](https://universaldependencies.org/treebanks/hi_hdtb/index.html)
  - [Parallel Universal Dependencies Treebank in Hindi](http://universaldependencies.org/treebanks/hi_pud/index.html) - A smaller part of the above-mentioned treebank.
- [ISI FIRE Stopwords List (Hindi and Bangla)](https://www.isical.ac.in/~fire/data/)
- [Peter Graham's Stopwords List](https://github.com/6/stopwords-json)
- [NLTK Corpus](https://www.nltk.org/book/ch02.html) 60k Words POS Tagged, Bangla, Hindi, Marathi, Telugu
- [Hindi Movie Reviews Dataset](https://github.com/goru001/nlp-for-hindi) ~1k Samples, 3 polarity classes
- [BBC News Hindi Dataset](https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1) 4.3k Samples, 14 classes
- [IIT Patna Hindi ABSA Dataset](https://github.com/pnisarg/ABSA) 5.4k Samples, 12 Domains, 4k aspect terms, aspect and sentence level polarity in 4 classes
- [Bangla ABSA](https://github.com/AtikRahman/Bangla_Datasets_ABSA) 5.5k Samples, 2 Domains, 10 aspect terms
- [IIT Patna Movie Review Sentiment Dataset](https://www.iitp.ac.in/~ai-nlp-ml/resources.html) 2k Samples, 3 polarity labels

#### Corpora/Datasets that need a login/access can be gained via email

- [SAIL 2015](http://amitavadas.com/SAIL/) Twitter and Facebook labelled sentiment samples in Hindi, Bengali, Tamil, Telugu.
- [IIT Bombay CFILT Resources](https://www.cfilt.iitb.ac.in/) - Sentiwordnet, parallel labelled corpora, sense-annotated corpora, and Marathi polarity-labelled corpus.
- [TDIL-IC aggregates a lot of useful resources and provides access to otherwise gated datasets](https://tdil-dc.in/index.php?option=com_catalogue&task=viewTools&id=83&lang=en)

### Language Models and Word Embeddings

- [Hindi2Vec](https://nirantk.com/hindi2vec/) and [nlp-for-hindi](https://github.com/goru001/nlp-for-hindi) ULMFIT style languge model
- [IIT Patna Bilingual Word Embeddings Hi-En](https://www.iitp.ac.in/~ai-nlp-ml/resources.html)
- [Fasttext word embeddings in a whole bunch of languages, trained on Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Hindi and Bengali Word2Vec](https://github.com/Kyubyong/wordvectors)
- [Hindi and Urdu Elmo Model](https://github.com/HIT-SCIR/ELMoForManyLangs)
- [Sanskrit Albert](https://huggingface.co/surajp/albert-base-sanskrit) Trained on Sanskrit Wikipedia and OSCAR corpus

### Libraries and Tooling

- [Multi-Task Deep Morphological Analyzer](https://github.com/Saurav0074/mt-dma) - deep morphological parser for Hindi and Urdu.
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) - tokenization, transliteration, MT helpers across 18 Indic languages.
- [SivaReddy's Dependency Parser (Python3 port)](https://github.com/CalmDownKarm/sivareddydependencyparser) - dependency parsing and POS tagging for Kannada, Hindi, and Telugu.
- [iNLTK](https://github.com/goru001/inltk) - NLP toolkit for Indic languages on PyTorch/Fastai.
- [AI4Bharat IndicNLP Suite](https://ai4bharat.iitm.ac.in/) - tools, datasets, and models across 22 Indic languages.

### Models and Embeddings

- [IndicBERT v2](https://github.com/AI4Bharat/IndicBERT) (2022-2024) - multilingual BERT for 23 Indic languages.
- [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (2023-2024) - high-quality MT for 22 Indic languages.
- [OpenHathi](https://huggingface.co/sarvamai/OpenHathi-7B-Hi-v0.1-Base) (Sarvam AI, 2023) - bilingual Hindi-English LLaMA continuation.
- [Airavata](https://huggingface.co/ai4bharat/Airavata) (2024) - instruction-tuned Hindi LLM.
- [Sarvam-1](https://www.sarvam.ai/blogs/sarvam-1) (2024) - multilingual LM trained from scratch on 10 Indic languages.
- [BharatGPT / Krutrim](https://www.olakrutrim.com/) (2024) - Indic-focused foundation models.

</details>

<details>
<summary>

### NLP in Indonesian

</summary>

[Back to Top](#contents)

### Libraries and Embeddings

- [bahasa](https://github.com/kangfend/bahasa) - natural language toolkit for Indonesian.
- [Indonesian Word Embedding](https://github.com/galuhsahid/indonesian-word-embedding)
- [Indonesian fastText](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.id.zip) trained on Wikipedia.

### Models

- [IndoBERT (IndoNLU)](https://github.com/indobenchmark/indonlu) - pretrained Indonesian LM with the IndoNLU benchmark suite.
- [IndoBERT (IndoLEM)](https://github.com/indolem/indolem) - alternative IndoBERT with the IndoLEM benchmark.
- [NusaCrowd / Cendol](https://github.com/IndoNLP/nusa-crowd) (2023-2024) - large-scale community datasets and Cendol instruction-tuned LMs for Indonesian and regional languages.
- [Sailor](https://github.com/sail-sg/sailor-llm) - open Southeast-Asian LMs covering Indonesian.
- [SEA-LION](https://github.com/aisingapore/sealion) (2024) - Singapore AI's open Southeast-Asian LM with strong Indonesian.

### Datasets

- Kompas and Tempo collections at [ILPS](http://ilps.science.uva.nl/resources/bahasa/)
- [PANL10N for PoS tagging](http://www.panl10n.net/english/outputs/Indonesia/UI/0802/UI-1M-tagged.zip): 39K sentences and 900K word tokens
- [IDN for PoS tagging](https://github.com/famrashel/idn-tagged-corpus): 10K sentences and 250K word tokens.
- [Indonesian Treebank](https://github.com/famrashel/idn-treebank) and [Universal Dependencies-Indonesian](https://github.com/UniversalDependencies/UD_Indonesian-GSD)
- [IndoSum](https://github.com/kata-ai/indosum) - text summarization and classification.
- [Wordnet-Bahasa](http://wn-msa.sourceforge.net/) - large, free, semantic dictionary.

</details>

<details>
<summary>

### NLP in Korean

</summary>

[Back to Top](#contents)

### Libraries

- [KoNLPy](http://konlpy.org) - Python package for Korean natural language processing.
- [Mecab (Korean)](https://eunjeon.blogspot.com/) - C++ library for Korean NLP.
- [KoalaNLP](https://koalanlp.github.io/koalanlp/) - Scala library for Korean NLP.
- [KoNLP](https://cran.r-project.org/package=KoNLP) - R package for Korean NLP.
- [kss](https://github.com/hyunwoongko/kss) - Korean sentence splitter.
- [Kiwi](https://github.com/bab2min/Kiwi) - fast Korean morphological analyzer.

### Models and Embeddings

- [KoBERT](https://github.com/SKTBrain/KoBERT) - Korean BERT from SKT.
- [KLUE-RoBERTa](https://github.com/KLUE-benchmark/KLUE) - models trained on the KLUE benchmark.
- [Polyglot-Ko](https://github.com/EleutherAI/polyglot) - open Korean LMs.
- [EXAONE 3.5](https://github.com/LG-AI-EXAONE) (LG, 2024) - bilingual Korean-English open LM family.
- [HyperCLOVA X](https://www.ncloud.com/product/aiService/clovaStudio) - Naver's Korean foundation model.

### Blogs and Tutorials

- [dsindex's blog](https://dsindex.github.io/)
- [Kangwon University's NLP course in Korean](http://cs.kangwon.ac.kr/~leeck/NLP/)

### Datasets

- [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus) - corpus from the Korea Advanced Institute of Science and Technology in Korean.
- [Naver Sentiment Movie Corpus in Korean](https://github.com/e9t/nsmc/)
- [Chosun Ilbo archive](http://srchdb1.chosun.com/pdf/i_archive/) - dataset in Korean from a major South Korean newspaper.
- [Chat data](https://github.com/songys/Chatbot_data) - chatbot data in Korean.
- [Petitions](https://github.com/akngs/petitions) - expired petition data from the Blue House National Petition Site.
- [Korean Parallel corpora](https://github.com/j-min/korean-parallel-corpora) - NMT dataset for Korean to French and Korean to English.
- [KorQuAD](https://korquad.github.io/) - Korean SQuAD dataset (v1.0 and v2.1) with Wiki HTML source.

</details>

<details>
<summary>

### NLP in Persian

</summary>

[Back to Top](#contents)

### Libraries

- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP toolkit.
- [Parsivar](https://github.com/ICTRC/Parsivar) - Persian language processing toolkit.
- [Perke](https://github.com/AlirezaTheH/perke) - Persian keyphrase extraction.
- [Perstem](https://github.com/jonsafari/perstem) - Persian stemmer, morphological analyzer, and partial POS tagger.
- [ParsiAnalyzer](https://github.com/NarimanN2/ParsiAnalyzer) - Persian analyzer for Elasticsearch.
- [virastar](https://github.com/aziz/virastar) - Persian text cleaning.

### Models

- [ParsBERT](https://github.com/hooshvare/parsbert) - Persian BERT.
- [PersianMind](https://huggingface.co/universitytehran/PersianMind-v1.0) (2023-2024) - Persian instruction-tuned LM.
- [Dorna](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct) (Part AI, 2024) - Llama-3-based Persian instruction model.

### Datasets

- [Bijankhan Corpus](https://dbrg.ut.ac.ir/بیژن%E2%80%8Cخان/) - tagged corpus suitable for Persian (Farsi) NLP research, ~2.6M manually tagged words across 40 POS tags.
- [Uppsala Persian Corpus (UPC)](https://sites.google.com/site/mojganserajicom/home/upc) - large freely available Persian corpus, 2.7M tokens annotated with 31 POS tags.
- [Large-Scale Colloquial Persian](http://hdl.handle.net/11234/1-3195) - LSCP: 120M sentences from 27M casual Persian tweets with dependency, POS, and sentiment annotations.
- [ArmanPersoNERCorpus](https://github.com/HaniehP/PersianNER) - 250K tokens, 7,682 sentences with NER tags in IOB format.
- [FarsiYar PersianNER](https://github.com/Text-Mining/Persian-NER) - ~25M tokens, ~1M Persian sentences from [Persian Wikipedia Corpus](https://github.com/Text-Mining/Persian-Wikipedia-Corpus).
- [PERLEX](http://farsbase.net/PERLEX.html) - first Persian dataset for relation extraction (translated SemEval-2010 Task 8).
- [Persian Syntactic Dependency Treebank](http://dadegan.ir/catalog/perdt) - 29,982 annotated sentences covering most verbs of the Persian valency lexicon.
- [Uppsala Persian Dependency Treebank (UPDT)](http://stp.lingfil.uu.se/~mojgan/UPDT.html) - dependency-based syntactically annotated corpus.
- [Hamshahri](https://dbrg.ut.ac.ir/hamshahri/) - standard reliable Persian text collection used at CLEF 2008-2009.

</details>

<details>
<summary>

### NLP in Polish

</summary>

[Back to Top](#contents)

- [Polish-NLP](https://github.com/ksopyla/awesome-nlp-polish) - curated list of resources dedicated to Polish NLP: models, tools, and datasets.

</details>

<details>
<summary>

### NLP in Portuguese

</summary>

[Back to Top](#contents)

- [Portuguese-nlp](https://github.com/ajdavidl/Portuguese-NLP) - curated list of Portuguese NLP resources and tools.

### Models

- [BERTimbau](https://github.com/neuralmind-ai/portuguese-bert) - BERT for Brazilian Portuguese.
- [Sabiá](https://huggingface.co/maritaca-ai) (Maritaca AI, 2023-2024) - Portuguese-focused open LMs.
- [Albertina](https://huggingface.co/PORTULAN) (PORTULAN, 2023-2024) - encoder-only Portuguese LMs for both PT-PT and PT-BR.

</details>

<details>
<summary>

### NLP in Spanish

</summary>

[Back to Top](#contents)

### Libraries

- [spanlp](https://github.com/jfreddypuentes/spanlp) - Python library to detect, censor, and clean profanity, hate speech, and bullying in Spanish, with data from 21 Spanish-speaking countries.

### Data

- [Columbian Political Speeches](https://github.com/dav009/LatinamericanTextResources)
- [Copenhagen Treebank](https://mbkromann.github.io/copenhagen-dependency-treebank/)
- [Spanish Billion Words Corpus with Word2Vec embeddings](https://github.com/crscardellino/sbwce)
- [Compilation of Spanish Unannotated Corpora](https://github.com/josecannete/spanish-unannotated-corpora)

### Models and Embeddings

- [BETO](https://github.com/dccuchile/beto) - BERT for Spanish.
- [RoBERTa-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne) - Spanish RoBERTa trained on the Spanish National Library corpus.
- [Latxa](https://github.com/hitz-zentroa/latxa) (2024) - open foundation LM for Basque, also covers Spanish.
- [Salamandra](https://huggingface.co/BSC-LT/salamandra-7b) (BSC, 2024) - multilingual LM with strong Spanish coverage from the Barcelona Supercomputing Center.
- [RigoChat](https://huggingface.co/IIC/RigoChat-7b-v2) (2024) - Spanish-instruction-tuned open model.
- [Spanish Word Embeddings (multiple methods/corpora)](https://github.com/dccuchile/spanish-word-embeddings)
- [Spanish fastText Embeddings](https://github.com/BotCenter/spanishWordEmbeddings)
- [Spanish sent2vec Sentence Embeddings](https://github.com/BotCenter/spanishSent2Vec)

</details>

<details>
<summary>

### NLP in Thai

</summary>

[Back to Top](#contents)

### Libraries

- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) - Thai NLP in Python.
- [JTCC](https://github.com/wittawatj/jtcc) - character cluster library in Java.
- [CutKum](https://github.com/pucktada/cutkum) - word segmentation with deep learning in TensorFlow.
- [Thai Language Toolkit](https://pypi.python.org/pypi/tltk/) - tokenization and POS tagging.
- [SynThai](https://github.com/KenjiroAI/SynThai) - word segmentation and POS tagging using deep learning.

### Models

- [WangchanBERTa](https://github.com/vistec-AI/thai2transformers) - pretrained Thai language model.
- [Typhoon](https://huggingface.co/scb10x) (SCB 10X, 2024) - open Thai LLM family.
- [OpenThaiGPT](https://huggingface.co/openthaigpt) (2023-2024) - open Thai instruction-tuned models.
- [Sailor](https://github.com/sail-sg/sailor-llm) - open Southeast-Asian LM family covering Thai.

### Data

- [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm) - text corpus with 5M words and word segmentation.
- [Prime Minister 29](https://github.com/PyThaiNLP/lexicon-thai/tree/master/thai-corpus/Prime%20Minister%2029) - dataset of speeches by the current Prime Minister of Thailand.

</details>

<details>
<summary>

### NLP in Ukrainian

</summary>

[Back to Top](#contents)

- [awesome-ukrainian-nlp](https://github.com/asivokon/awesome-ukrainian-nlp) - curated list of Ukrainian NLP datasets, models, etc.
- [UkrainianLT](https://github.com/Helsinki-NLP/UkrainianLT) - curated list focused on machine translation and speech processing.

</details>

<details>
<summary>

### NLP in Urdu

</summary>

[Back to Top](#contents)

### Libraries

- [urduhack](https://github.com/urduhack/urduhack) - NLP library for Urdu.

### Datasets

- [Collection of Urdu datasets](https://github.com/mirfan899/Urdu) - POS, NER, and other NLP tasks.

</details>

<details>
<summary>

### NLP in Vietnamese

</summary>

[Back to Top](#contents)

### Libraries

- [underthesea](https://github.com/undertheseanlp/underthesea) - Vietnamese NLP toolkit.
- [vn.vitk](https://github.com/phuonglh/vn.vitk) - Vietnamese text processing toolkit.
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) - Vietnamese NLP toolkit.
- [pyvi](https://github.com/trungtv/pyvi) - Python Vietnamese core NLP toolkit.
- [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) - on-device Vietnamese text-to-speech with voice cloning.

### Models and Embeddings

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - pretrained LM for Vietnamese.
- [BARTpho](https://github.com/VinAIResearch/BARTpho) - sequence-to-sequence pretrained model for Vietnamese.
- [PhoGPT](https://github.com/VinAIResearch/PhoGPT) (VinAI, 2023-2024) - open generative LM for Vietnamese.
- [Vistral](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) (2024) - Mistral-based Vietnamese chat model.
- [Sailor](https://github.com/sail-sg/sailor-llm) (2024) - open multilingual LM family covering Vietnamese, Thai, Indonesian, and other Southeast Asian languages.

### Data

- [Vietnamese Treebank](https://vlsp.hpda.vn/demo/?page=resources&lang=en) - 10K sentences for the constituency parsing task.
- [BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf) - Vietnamese dependency treebank.
- [UD_Vietnamese](https://github.com/UniversalDependencies/UD_Vietnamese-VTB) - Vietnamese Universal Dependency Treebank.
- [VIVOS](https://ailab.hcmus.edu.vn/vivos/) - free Vietnamese speech corpus, 15 hours of recorded speech (HCMUS AILab).
- [VNTQcorpus(big).txt](http://viet.jnlp.org/download-du-lieu-tu-vung-corpus) - 1.75M news sentences.
- [ViText2SQL](https://github.com/VinAIResearch/ViText2SQL) - Vietnamese Text-to-SQL semantic parsing dataset (EMNLP-2020 Findings).
- [EVB Corpus](https://github.com/qhungngo/EVBCorpus) - 20M words across 15 bilingual books, 100 parallel English-Vietnamese texts, 250 parallel law texts, 5K news articles, and 2K film subtitles.

</details>

### Other Languages

- Russian: [pymorphy2](https://github.com/kmike/pymorphy2) - a good pos-tagger for Russian
- Asian Languages: Thai, Lao, Chinese, Japanese, and Korean [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html) implementation in ElasticSearch
- Ancient Languages: [CLTK](https://github.com/cltk/cltk): The Classical Language Toolkit is a Python library and collection of texts for doing NLP in ancient languages
- Hebrew: [NLPH_Resources](https://github.com/NLPH/NLPH_Resources) - A collection of papers, corpora and linguistic resources for NLP in Hebrew

[Back to Top](#contents)

## See Also

Adjacent curated lists for topics out of scope here:

- [awesome-llm](https://github.com/Hannibal046/Awesome-LLM) - general-purpose large language model resources.
- [awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) - generative AI across modalities.
- [awesome-rag](https://github.com/Danielskry/Awesome-RAG) - retrieval-augmented generation systems and tooling.
- [awesome-prompt-engineering](https://github.com/promptslab/Awesome-Prompt-Engineering) - prompting techniques and template libraries.
- [awesome-mlops](https://github.com/visenger/awesome-mlops) - production ML, including LLM serving.

## Citation

If you find this repository useful, please consider citing this list:

```bibtex
@misc{awesome-nlp,
  title  = {Awesome NLP},
  author = {Kim, Keon Woo},
  year   = {2018},
  url    = {https://github.com/keon/awesome-nlp},
  note   = {GitHub repository}
}
```

## License
[License](./LICENSE) - CC0
