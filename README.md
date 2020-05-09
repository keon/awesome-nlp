# awesome-nlp 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources dedicated to Natural Language Processing

![Awesome NLP Logo](/images/logo.jpg)

Read this in [English](./README.md), [Traditional Chinese](./README-ZH-TW.md) 

_Please read the [contribution guidelines](contributing.md) before contributing. Please add your favourite NLP resource by raising a [pull request](https://github.com/keonkim/awesome-nlp/pulls)_

## Contents

* [Research Summaries and Trends](#research-summaries-and-trends)
* [Prominent NLP Research Labs](#prominent-nlp-research-labs)
* [Tutorials](#tutorials)
  * [Reading Content](#reading-content)
  * [Videos and Courses](#videos-and-online-courses)
  * [Books](#books)
* [Libraries](#libraries)
  * [Node.js](#user-content-node-js)
  * [Python](#user-content-python)
  * [C++](#user-content-c++)
  * [Java](#user-content-java)
  * [Kotlin](#user-content-kotlin)
  * [Scala](#user-content-scala)
  * [R](#user-content-r)
  * [Clojure](#user-content-clojure)
  * [Ruby](#user-content-ruby)
  * [Rust](#user-content-rust)
* [Services](#services)
* [Annotation Tools](#annotation-tools)
* [Datasets](#datasets)
* [NLP in Korean](#nlp-in-korean)
* [NLP in Arabic](#nlp-in-arabic)
* [NLP in Chinese](#nlp-in-chinese)
* [NLP in German](#nlp-in-german)
* [NLP in Polish](#nlp-in-polish)
* [NLP in Spanish](#nlp-in-spanish)
* [NLP in Indic Languages](#nlp-in-indic-languages)
* [NLP in Thai](#nlp-in-thai)
* [NLP in Danish](#nlp-in-danish)
* [NLP in Vietnamese](#nlp-in-vietnamese)
* [NLP for Dutch](#nlp-for-dutch)
* [NLP in Indonesian](#nlp-in-indonesian)
* [Other Languages](#other-languages)
* [Credits](#credits)

## Research Summaries and Trends

* [NLP-Overview](https://nlpoverview.com/) is an up-to-date overview of deep learning techniques applied to NLP, including theory, implementations, applications, and state-of-the-art results. This is a great Deep NLP Introduction for researchers. 
* [NLP-Progress](https://nlpprogress.com/) tracks the progress in Natural Language Processing, including the datasets and the current state-of-the-art for the most common NLP tasks
* [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
* [ACL 2018 Highlights: Understanding Representation and Evaluation in More Challenging Settings](http://ruder.io/acl-2018-highlights/)
* [Four deep learning trends from ACL 2017. Part One: Linguistic Structure and Word Embeddings](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-1.html)
* [Four deep learning trends from ACL 2017. Part Two: Interpretability and Attention](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-2.html)
* [Highlights of EMNLP 2017: Exciting Datasets, Return of the Clusters, and More!](http://blog.aylien.com/highlights-emnlp-2017-exciting-datasets-return-clusters/)
* [Deep Learning for Natural Language Processing (NLP): Advancements & Trends](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)
* [Survey of the State of the Art in Natural Language Generation](https://arxiv.org/abs/1703.09902)

## Prominent NLP Research Labs
[Back to Top](#contents)

* [The Berkeley NLP Group](http://nlp.cs.berkeley.edu/index.shtml) - Notable contributions include a tool to reconstruct long dead languages, referenced [here](https://www.bbc.com/news/science-environment-21427896) and by taking corpora from 637 languages currently spoken in Asia and the Pacific and recreating their descendant.
* [Language Technologies Institute, Carnegie Mellon University](http://www.cs.cmu.edu/~nasmith/nlp-cl.html) - Notable projects include [Avenue Project](http://www.cs.cmu.edu/~avenue/), a syntax driven machine translation system for endangered languages like Quechua and Aymara and previously, [Noah's Ark](http://www.cs.cmu.edu/~ark/) which created [AQMAR](http://www.cs.cmu.edu/~ark/AQMAR/) to improve NLP tools for Arabic.
* [NLP research group, Columbia University](http://www1.cs.columbia.edu/nlp/index.cgi) - Responsible for creating BOLT ( interactive error handling for speech translation systems) and an un-named project to characterize laughter in dialogue.
* [The Center or Language and Speech Processing, John Hopkins University](http://clsp.jhu.edu/) - Recently in the news for developing speech recognition software to create a diagnostic test or Parkinson's Disease, [here](https://www.clsp.jhu.edu/2019/03/27/speech-recognition-software-and-machine-learning-tools-are-being-used-to-create-diagnostic-test-for-parkinsons-disease/#.XNFqrIkzYdU).
* [Computational Linguistics and Information Processing Group, University of Maryland](https://wiki.umiacs.umd.edu/clip/index.php/Main_Page) - Notable contributions include [Human-Computer Cooperation or Word-by-Word Question Answering](http://www.umiacs.umd.edu/~jbg/projects/IIS-1652666) and modeling development of phonetic representations. 
* [Penn Natural Language Processing, University of Pennsylvania](http://nlp.cis.upenn.edu/index.php)- Famous for creating the [Penn Treebank](http://www.cis.upenn.edu/~treebank/).
* [The Stanford Nautral Language Processing Group](https://nlp.stanford.edu/)- One of the top NLP research labs in the world, notable for creating [Stanford CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) and their [coreference resolution system](https://nlp.stanford.edu/software/dcoref.shtml)


## Tutorials
[Back to Top](#contents)

### Reading Content

General Machine Learning

* [Machine Learning 101](https://docs.google.com/presentation/d/1kSuQyW5DTnkVaZEjGYCkfOxvzCqGEFzWBy4e9Uedd9k/edit?usp=sharing) from Google's Senior Creative Engineer explains Machine Learning for engineer's and executives alike
* [AI Playbook](https://aiplaybook.a16z.com/) - a16z AI playbook is a great link to forward to your managers or content for your presentations
* [Ruder's Blog](http://ruder.io/#open) by [Sebastian Ruder](https://twitter.com/seb_ruder) for commentary on the best of NLP Research
* [How To Label Data](https://www.lighttag.io/how-to-label-data/) guide to managing larger linguistic annotation projects

Introductions and Guides to NLP

* [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [NLP in Python](http://github.com/NirantK/nlp-python-deep-learning) - Collection of Github notebooks 
* [Natural Language Processing: An Introduction](https://academic.oup.com/jamia/article/18/5/544/829676) - Oxford
* [Deep Learning for NLP with Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
* [Hands-On NLTK Tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) - NLTK Tutorials, Jupyter notebooks 
* [Train a new language model from scratch](https://huggingface.co/blog/how-to-train) - Hugging Face 🤗
* [The Super Duper NLP Repo (SDNLPR)](https://notebooks.quantumstat.com/): Collection of Colab notebooks covering a wide array of NLP task implementations.

Blogs and Newsletters

* [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/) and [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Natural Language Processing](https://nlpers.blogspot.com/) by Hal Daumé III
* [arXiv: Natural Language Processing (Almost) from Scratch](https://arxiv.org/pdf/1103.0398.pdf)
* [Karpathy's The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
* [Machine Learning Mastery: Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing)
* [Visual Paper Summary: ALBERT (A Lite BERT)](https://amitness.com/2020/02/albert-visual-summary/)

### Videos and Online Courses
[Back to Top](#contents)

* [Deep Natural Language Processing](https://github.com/oxford-cs-deepnlp-2017/lectures) - Lectures series from Oxford
* [Deep Learning for Natural Language Processing (cs224-n)](https://web.stanford.edu/class/cs224n/) - Richard Socher and Christopher Manning's Stanford Course
* [Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/) - Carnegie Mellon Language Technology Institute there
* [Deep NLP Course](https://github.com/yandexdataschool/nlp_course) by Yandex Data School, covering important ideas from text embedding to machine translation including sequence modeling, language models and so on. 
* [fast.ai Code-First Intro to Natural Language Processing](https://www.fast.ai/2019/07/08/fastai-nlp/) - This covers a blend of traditional NLP topics (including regex, SVD, naive bayes, tokenization) and recent neural network approaches (including RNNs, seq2seq, GRUs, and the Transformer), as well as addressing urgent ethical issues, such as bias and disinformation. Find the Jupyter Notebooks [here](https://github.com/fastai/course-nlp)   


### Books

* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - free, by Prof. Dan Jurafsy
* [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class) - free, NLP notes by Dr. Jacob Eisenstein at GeorgiaTech 
* [NLP with PyTorch](https://github.com/joosthub/PyTorchNLPBook) - Brian & Delip Rao 
* [Text Mining in R](https://www.tidytextmining.com)
* [Natural Language Processing with Python](https://www.nltk.org/book/)

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

  - [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of [Natural Language Toolkit (NLTK)](https://www.nltk.org/) and [Pattern](https://github.com/clips/pattern), and plays nicely with both :+1:
  - [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython :+1:
    - [textacy](https://github.com/chartbeat-labs/textacy) - Higher level NLP built on spaCy
  - [gensim](https://radimrehurek.com/gensim/index.html) - Python library to conduct unsupervised semantic modelling from plain text :+1:
  - [scattertext](https://github.com/JasonKessler/scattertext) - Python library to produce d3 visualizations of how language differs between corpora
  - [GluonNLP](https://github.com/dmlc/gluon-nlp) - A deep learning toolkit for NLP, built on MXNet/Gluon, for research prototyping and industrial deployment of state-of-the-art models on a wide range of NLP tasks.
  - [AllenNLP](https://github.com/allenai/allennlp) - An NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
  - [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP research toolkit designed to support rapid prototyping with better data loaders, word vector loaders, neural network layer representations, common NLP metrics such as BLEU
  - [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
  - [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
  - [jPTDP](https://github.com/datquocnguyen/jPTDP) - A toolkit for joint part-of-speech (POS) tagging and dependency parsing. jPTDP provides pre-trained models for 40+ languages.
  - [BigARTM](https://github.com/bigartm/bigartm) - a fast library for topic modelling
  - [Snips NLU](https://github.com/snipsco/snips-nlu) - A production ready library for intent parsing
  - [Chazutsu](https://github.com/chakki-works/chazutsu) - A library for downloading&parsing standard NLP research datasets
  - [Word Forms](https://github.com/gutfeeling/word_forms) - Word forms can accurately generate all possible forms of an English word
  - [Multilingual Latent Dirichlet Allocation (LDA)](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA) - A multilingual and extensible document clustering pipeline
  - [NLP Architect](https://github.com/NervanaSystems/nlp-architect) - A library for exploring the state-of-the-art deep learning topologies and techniques for NLP and NLU
  - [Flair](https://github.com/zalandoresearch/flair) - A very simple framework for state-of-the-art multilingual NLP built on PyTorch. Includes BERT, ELMo and Flair embeddings.
  - [Kashgari](https://github.com/BrikerMan/Kashgari) - Simple, Keras-powered multilingual NLP framework, allows you to build your models in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS) and text classification tasks. Includes BERT and word2vec embedding.
  - [FARM](https://github.com/deepset-ai/FARM) - FARM makes cutting-edge transfer learning simple and helps you to leverage pretrained language models for your own NLP tasks.
  - [Rita DSL](https://github.com/zaibacu/rita-dsl) - a DSL, loosely based on [RUTA on Apache UIMA](https://uima.apache.org/ruta.html). Allows to define language patterns (rule-based NLP) which are then translated into [spaCy](https://spacy.io/), or if you prefer less features and lightweight - regex patterns.
  - [Transformers](https://github.com/huggingface/transformers) - Natural Language Processing for TensorFlow 2.0 and PyTorch.
  - [Tokenizers](https://github.com/huggingface/tokenizers) - Tokenizers optimized for Research and Production.
  - [fairSeq](https://github.com/pytorch/fairseq) Facebook AI Research implementations of SOTA seq2seq models in Pytorch. 

- <a id="c++">**C++** - C++ Libraries</a> | [Back to Top](#contents)
  - [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
  - [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
  - [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data.
  - [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  - [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  - [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
  - [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia/)
  - [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
  - [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
  - [Mecab (Japanese)](https://taku910.github.io/mecab/)
  - [Moses](http://statmt.org/moses/)
  - [StarSpace](https://github.com/facebookresearch/StarSpace) - a library from Facebook for creating embeddings of word-level, paragraph-level, document-level and for text classification

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

- <a id="clojure">**Clojure**</a> | [Back to Top](#contents)
  - [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
  - [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript
  - [postagga](https://github.com/fekr/postagga) - A library to parse natural language in Clojure and ClojureScript

- <a id="ruby">**Ruby**</a> | [Back to Top](#contents)
  - Kevin Dias's [A collection of Natural Language Processing (NLP) Ruby libraries, tools and software](https://github.com/diasks2/ruby-nlp)
  - [Practical Natural Language Processing done in Ruby](https://github.com/arbox/nlp-with-ruby)

- <a id="rust">**Rust**</a>
  - [whatlang](https://github.com/greyblake/whatlang-rs) — Natural language recognition library based on trigrams
  - [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) - A production ready library for intent parsing
  - [rust-bert](https://github.com/guillaume-be/rust-bert) - Ready-to-use NLP pipelines and Transformer-based models

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

### Annotation Tools

- [GATE](https://gate.ac.uk/overview.html) - General Architecture and Text Engineering is 15+ years old, free and open source
- [Anafora](https://github.com/weitechen/anafora) is free and open source, web-based raw text annotation tool
- [brat](https://brat.nlplab.org/) - brat rapid annotation tool is an online environment for collaborative text annotation
- [doccano](https://github.com/chakki-works/doccano) - doccano is free, open-source, and provides annotation features for text classification, sequence labeling and sequence to sequence
- [tagtog](https://www.tagtog.net/), team-first web tool to find, create, maintain, and share datasets - costs $
- [prodigy](https://prodi.gy/) is an annotation tool powered by active learning, costs $
- [LightTag](https://lighttag.io) - Hosted and managed text annotation tool for teams, costs $
- [rstWeb](https://corpling.uis.georgetown.edu/rstweb/info/) - open source local or online tool for discourse tree annotations
- [GitDox](https://corpling.uis.georgetown.edu/gitdox/) - open source server annotation tool with GitHub version control and validation for XML data and collaborative spreadsheet grids
- [Label Studio](https://www.heartex.ai/) - Hosted and managed text annotation tool for teams, freemium based, costs $

## Techniques

### Text Embeddings

#### Word Embeddings

- Thumb Rule: **fastText >> GloVe > word2vec**

- [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - [implementation](https://code.google.com/archive/p/word2vec/) - [explainer blog](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [glove](https://nlp.stanford.edu/pubs/glove.pdf) - [explainer blog](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/)
- fasttext - [implementation](https://github.com/facebookresearch/fastText) - [paper](https://arxiv.org/abs/1607.04606) - [explainer blog](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)

#### Sentence and Language Model Based Word Embeddings

[Back to Top](#contents)

- ElMo - [Deep Contextualized Word Represenations](https://arxiv.org/abs/1802.05365) - [PyTorch implmentation](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) - [TF Implementation](https://github.com/allenai/bilm-tf)
- ULMFiT - [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder
- InferSent - [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) by facebook
- CoVe - [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
- Pargraph vectors - from [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf). See [doc2vec tutorial at gensim](https://rare-technologies.com/doc2vec-tutorial/)
- [sense2vec](https://arxiv.org/abs/1511.06388) - on word sense disambiguation
- [Skip Thought Vectors](https://arxiv.org/abs/1506.06726) - word representation method
- [Adaptive skip-gram](https://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties
- [Sequence to Sequence Learning](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - word vectors for machine translation

### Question Answering and Knowledge Extraction

[Back to Top](#contents)

- [DrQA](https://github.com/facebookresearch/DrQA) - Open Domain Question Answering work by Facebook Research on Wikipedia data
- [Document-QA](https://github.com/allenai/document-qa) - Simple and Effective Multi-Paragraph Reading Comprehension by AllenAI
- [Template-Based Information Extraction without the Templates](https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
- [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)

## Datasets

[Back to Top](#contents)

- [nlp-datasets](https://github.com/niderhoff/nlp-datasets) great collection of nlp datasets

## Multilingual NLP Frameworks

[Back to Top](#contents)

- [UDPipe](https://github.com/ufal/udpipe) is a trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U files. Primarily written in C++, offers a fast and reliable solution for multilingual NLP processing.
- [NLP-Cube](https://github.com/adobe/NLP-Cube) : Natural Language Processing Pipeline - Sentence Splitting, Tokenization, Lemmatization, Part-of-speech Tagging and Dependency Parsing. New platform, written in Python with Dynet 2.0. Offers standalone (CLI/Python bindings) and server functionality (REST API).

## NLP in Korean

[Back to Top](#contents)

### Libraries

- [KoNLPy](http://konlpy.org) - Python package for Korean natural language processing.
- [Mecab (Korean)](https://eunjeon.blogspot.com/) - C++ library for Korean NLP
- [KoalaNLP](https://koalanlp.github.io/koalanlp/) - Scala library for Korean Natural Language Processing.
- [KoNLP](https://cran.r-project.org/package=KoNLP) - R package for Korean Natural language processing

### Blogs and Tutorials

- [dsindex's blog](https://dsindex.github.io/)
- [Kangwon University's NLP course in Korean](http://cs.kangwon.ac.kr/~leeck/NLP/)

### Datasets

- [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus) - A corpus from the Korea Advanced Institute of Science and Technology in Korean.
- [Naver Sentiment Movie Corpus in Korean](https://github.com/e9t/nsmc/)
- [Chosun Ilbo archive](http://srchdb1.chosun.com/pdf/i_archive/) - dataset in Korean from one of the major newspapers in South Korea, the Chosun Ilbo.
- [Chat data](https://github.com/songys/Chatbot_data) - Chatbot data in Korean
- [Petitions](https://github.com/akngs/petitions) - Collect expired petition data from the Blue House National Petition Site.
- [Korean Parallel corpora](https://github.com/j-min/korean-parallel-corpora) - Neural Machine Translation(NMT) Dataset for **Korean to French** & **Korean to English**
- [KorQuAD](https://korquad.github.io/) - Korean SQuAD dataset with Wiki HTML source. Mentions both v1.0 and v2.1 at the time of adding to Awesome NLP

## NLP in Arabic

[Back to Top](#contents)

### Libraries

- [goarabic](https://github.com/01walid/goarabic) - Go package for Arabic text processing
- [jsastem](https://github.com/ejtaal/jsastem) - Javascript for Arabic stemming
- [PyArabic](https://pypi.org/project/PyArabic/) - Python libraries for Arabic
- [RFTokenizer](https://github.com/amir-zeldes/RFTokenizer) - trainable Python segmenter for Arabic, Hebrew and Coptic

### Datasets

- [Multidomain Datasets](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces) - Largest Available Multi-Domain Resources for Arabic Sentiment Analysis
- [LABR](https://github.com/mohamedadaly/labr) - LArge Arabic Book Reviews dataset
- [Arabic Stopwords](https://github.com/mohataher/arabic-stop-words) - A list of Arabic stopwords from various resources

## NLP in Chinese

[Back to Top](#contents)

### Libraries

- [jieba](https://github.com/fxsjy/jieba#jieba-1) - Python package for Words Segmentation Utilities in Chinese
- [SnowNLP](https://github.com/isnowfy/snownlp) - Python package for Chinese NLP
- [FudanNLP](https://github.com/FudanNLP/fnlp) - Java library for Chinese text processing

### Anthology
- [funNLP](https://github.com/fighting41love/funNLP) - Collection of NLP tools and resources mainly for Chinese

## NLP in German

- [German-NLP](https://github.com/adbar/German-NLP) - Curated list of open-access/open-source/off-the-shelf resources and tools developed with a particular focus on German

## NLP in Polish

- [Polish-NLP](https://github.com/ksopyla/awesome-nlp-polish) - A curated list of resources dedicated to Natural Language Processing (NLP) in polish. Models, tools, datasets.

## NLP in Spanish

[Back to Top](#contents)

### Data

- [Columbian Political Speeches](https://github.com/dav009/LatinamericanTextResources)
- [Copenhagen Treebank](https://mbkromann.github.io/copenhagen-dependency-treebank/)
- [Spanish Billion words corpus with Word2Vec embeddings](https://github.com/crscardellino/sbwce)
- [Compilation of Spanish Unannotated Corpora](https://github.com/josecannete/spanish-unannotated-corpora)

### Word and Sentence Embeddings
- [Spanish Word Embeddings Computed with Different Methods and from Different Corpora](https://github.com/dccuchile/spanish-word-embeddings)
- [Spanish Word Embeddings Computed from Large Corpora and Different Sizes Using fastText](https://github.com/BotCenter/spanishWordEmbeddings)
- [Spanish Sentence Embeddings Computed from Large Corpora Using sent2vec](https://github.com/BotCenter/spanishSent2Vec)
- [Beto - BERT for Spanish](https://github.com/dccuchile/beto)


## NLP in Indic languages

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
- [IIT Bombay NLP Resources](http://www.cfilt.iitb.ac.in/Sentiment_Analysis_Resources.html) Sentiwordnet, Movie and Tourism parallel labelled corpora, polarity labelled sense annotated corpus, Marathi polarity labelled corpus.
- [TDIL-IC aggregates a lot of useful resources and provides access to otherwise gated datasets](https://tdil-dc.in/index.php?option=com_catalogue&task=viewTools&id=83&lang=en)

### Language Models and Word Embeddings

- [Hindi2Vec](https://nirantk.com/hindi2vec/) and [nlp-for-hindi](https://github.com/goru001/nlp-for-hindi) ULMFIT style languge model
- [IIT Patna Bilingual Word Embeddings Hi-En](https://www.iitp.ac.in/~ai-nlp-ml/resources.html)
- [Fasttext word embeddings in a whole bunch of languages, trained on Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Hindi and Bengali Word2Vec](https://github.com/Kyubyong/wordvectors)
- [Hindi and Urdu Elmo Model](https://github.com/HIT-SCIR/ELMoForManyLangs)

### Libraries and Tooling

- [Multi-Task Deep Morphological Analyzer](https://github.com/Saurav0074/mt-dma) Deep Network based Morphological Parser for Hindi and Urdu
- [Anoop Kunchukuttan](https://github.com/anoopkunchukuttan/indic_nlp_library) 18 Languages, whole host of features from tokenization to translation
- [SivaReddy's Dependency Parser](http://sivareddy.in/downloads) Dependency Parser and Pos Tagger for Kannada, Hindi and Telugu. [Python3 Port](https://github.com/CalmDownKarm/sivareddydependencyparser)
- [iNLTK](https://github.com/goru001/inltk) - A Natural Language Toolkit for Indic Languages (Indian subcontinent languages) built on top of Pytorch/Fastai, which aims to provide out of the box support for common NLP tasks.

## NLP in Thai

[Back to Top](#contents)

### Libraries

- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) - Thai NLP in Python Package
- [JTCC](https://github.com/wittawatj/jtcc) - A character cluster library in Java
- [CutKum](https://github.com/pucktada/cutkum) - Word segmentation with deep learning in TensorFlow
- [Thai Language Toolkit](https://pypi.python.org/pypi/tltk/) - Based on a paper by Wirote Aroonmanakun in 2002 with included dataset
- [SynThai](https://github.com/KenjiroAI/SynThai) - Word segmentation and POS tagging using deep learning in Python

### Data

- [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm) - A text corpus with 5 million words with word segmentation
- [Prime Minister 29](https://github.com/PyThaiNLP/lexicon-thai/tree/master/thai-corpus/Prime%20Minister%2029) - Dataset containing speeches of the current Prime Minister of Thailand

## NLP in Danish 

- [Named Entity Recognition for Danish](https://github.com/ITUnlp/daner)
- [DaNLP](https://github.com/alexandrainst/danlp) - NLP resources in Danish
- [Awesome Danish](https://github.com/fnielsen/awesome-danish) - A curated list of awesome resources for Danish language technology

## NLP in Vietnamese 

### Libraries

- [underthesea](https://github.com/undertheseanlp/underthesea) - Vietnamese NLP Toolkit
- [vn.vitk](https://github.com/phuonglh/vn.vitk) - A Vietnamese Text Processing Toolkit
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) - A Vietnamese natural language processing toolkit

### Data

- [Vietnamese treebank](https://vlsp.hpda.vn/demo/?page=resources&lang=en) - 10,000 sentences for the constituency parsing task
- [BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf) - a Vietnamese Dependency Treebank
- [UD_Vietnamese](https://github.com/UniversalDependencies/UD_Vietnamese-VTB) - Vietnamese Universal Dependency Treebank
- [VIVOS](https://ailab.hcmus.edu.vn/vivos/) - a free Vietnamese speech corpus consisting of 15 hours of recording speech by AILab
- [VNTQcorpus(big).txt](http://viet.jnlp.org/download-du-lieu-tu-vung-corpus) - 1.75 million sentences in news

## NLP for Dutch

[Back to Top](#contents)

- [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
- [SimpleNLG_NL](https://github.com/rfdj/SimpleNLG-NL) - Dutch surface realiser used for Natural Language Generation in Dutch, based on the SimpleNLG implementation for English and French.

## NLP in Indonesian 

### Datasets
- Kompas and Tempo collections at [ILPS](http://ilps.science.uva.nl/resources/bahasa/)
- [PANL10N for PoS tagging](http://www.panl10n.net/english/outputs/Indonesia/UI/0802/UI-1M-tagged.zip): 39K sentences and 900K word tokens
- [IDN for PoS tagging](https://github.com/famrashel/idn-tagged-corpus): This corpus contains 10K sentences and 250K word tokens
- [Indonesian Treebank](https://github.com/famrashel/idn-treebank) and [Universal Dependencies-Indonesian](https://github.com/UniversalDependencies/UD_Indonesian-GSD)
- [IndoSum](https://github.com/kata-ai/indosum) for text summarization and classification both
- [Wordnet-Bahasa](http://wn-msa.sourceforge.net/) - large, free, semantic dictionary

### Libraries & Embedding
- Natural language toolkit [bahasa](https://github.com/kangfend/bahasa)
- [Indonesian Word Embedding](https://github.com/galuhsahid/indonesian-word-embedding)
- Pretrained [Indonesian fastText Text Embedding](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.id.zip) trained on Wikipedia

## NLP in Urdu

### Datasets
- [Collection of Urdu datasets](https://github.com/mirfan899/Urdu) for POS, NER and NLP tasks

### Libraries
- [Natural Language Processing library](https://github.com/urduhack/urduhack) for ( 🇵🇰)Urdu language

## Other Languages 

- Russian: [pymorphy2](https://github.com/kmike/pymorphy2) - a good pos-tagger for Russian
- Asian Languages: Thai, Lao, Chinese, Japanese, and Korean [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html) implementation in ElasticSearch
- Ancient Languages: [CLTK](https://github.com/cltk/cltk): The Classical Language Toolkit is a Python library and collection of texts for doing NLP in ancient languages
- Hebrew: [NLPH_Resources](https://github.com/NLPH/NLPH_Resources) - A collection of papers, corpora and linguistic resources for NLP in Hebrew

[Back to Top](#contents)

[Credits](./CREDITS.md) for initial curators and sources

## License
[License](./LICENSE) - CC0
