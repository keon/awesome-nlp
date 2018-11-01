# awesome-nlp [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of resources dedicated to Natural Language Processing

![Awesome NLP Logo](/images/logo.jpg)

**Maintainers** - [Keon](https://github.com/keon), [Martin](https://github.com/outpark), [Nirant](https://github.com/NirantK), [Dhruv](https://github.com/the-ethan-hunt)

_Please read the [contribution guidelines](contributing.md) before contributing. Please feel free to create [pull requests](https://github.com/keonkim/awesome-nlp/pulls)._

## Contents

- [Research Summaries and Trends](#research-summaries-and-trends)
- [Tutorials](#tutorials)
  - [Reading Content](#reading-content)
  - [Videos and Courses](#videos-and-online-courses)
  - [Books](#books)
- [Libraries](#libraries)
  - [Node.js](#user-content-node-js)
  - [Python](#user-content-python)
  - [C++](#user-content-c++)
  - [Java](#user-content-java)
  - [Scala](#user-content-scala)
  - [R](#user-content-R)
  - [Clojure](#user-content-clojure)
  - [Ruby](#user-content-ruby)
  - [Rust](#user-content-rust)
- [Services](#services)
- [Annotation Tools](#annotation-tools)
- [Datasets](#datasets)
- [Implementations of various models](#implementations-of-various-models)
- [NLP in Korean](#nlp-in-korean)
- [NLP in Arabic](#nlp-in-arabic)
- [NLP in Chinese](#nlp-in-chinese)
- [NLP in German](#nlp-in-german)
- [NLP in Spanish](#nlp-in-spanish)
- [NLP in Indic Languages](#nlp-in-indic-languages)
- [NLP in Thai](#nlp-in-thai)
- [NLP in Vietnamese](#nlp-in-vietnamese)
- [Other Languages](#other-languages)
- [Credits](#credits)

## Research Summaries and Trends

- [NLP-Overview](https://nlpoverview.com/) is an up-to-date overview of deep learning techniques applied to NLP, including theory, implementations, applications, and state-of-the-art results. This is a great Deep NLP Introduction for researchers. 
- [NLP-Progress](https://nlpprogress.com/) tracks the progress in Natural Language Processing, including the datasets and the current state-of-the-art for the most common NLP tasks
- [Four deep learning trends from ACL 2017. Part One: Linguistic Structure and Word Embeddings](http://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-1.html)
- [Four deep learning trends from ACL 2017. Part Two: Interpretability and Attention](http://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-2.html)
- [Highlights of EMNLP 2017: Exciting Datasets, Return of the Clusters, and More!](http://blog.aylien.com/highlights-emnlp-2017-exciting-datasets-return-clusters/)
- [Deep Learning for Natural Language Processing (NLP): Advancements & Trends](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)
- [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
- [ACL 2018 Highlights: Understanding Representation and Evaluation in More Challenging Settings](http://ruder.io/acl-2018-highlights/)
- [Survey of the State of the Art in Natural Language Generation](https://arxiv.org/abs/1703.09902)

## Tutorials
[Back to Top](#contents)

### Reading Content

General Machine Learning

- Jason's [Machine Learning 101](https://docs.google.com/presentation/d/1kSuQyW5DTnkVaZEjGYCkfOxvzCqGEFzWBy4e9Uedd9k/edit?usp=sharing) from Google's Senior Creative Engineer explains Machine Learning for engineer's and executives alike
- a16z [AI Playbook](http://aiplaybook.a16z.com/) is a great link to forward to your managers or content for your presentations
- [Machine Learning Blog](https://bmcfee.github.io/#home) by Brian McFee
- [Ruder's Blog](http://ruder.io/#open) by [Sebastian Ruder](https://twitter.com/seb_ruder) for commentary on the best of NLP Research

Introductions and Guides to NLP

- Ultimate Guide to [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
- [Introduction to NLP at Hackernoon](https://hackernoon.com/learning-ai-if-you-suck-at-math-p7-the-magic-of-natural-language-processing-f3819a689386) is for people who suck at math - in their own words
- [NLP Tutorial by Vik Paruchari](http://www.vikparuchuri.com/blog/natural-language-processing-tutorial/)
- [Natural Language Processing: An Introduction](http://jamia.oxfordjournals.org/content/18/5/544.short) by Oxford
- [Deep Learning for NLP with Pytorch](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
- [Hands-On NLTK Tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) - The hands-on NLTK tutorial in the form of Jupyter notebooks

Blogs and Newsletters

- [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [Natural Language Processing Blog](http://nlpers.blogspot.ch/) by Hal Daumé III
- [Tutorials by Radim Řehůřek](https://radimrehurek.com/gensim/tutorial.html) on using Python and [gensim](https://radimrehurek.com/gensim/index.html) to process language corpora
- [arXiv: Natural Language Processing (Almost) from Scratch](http://arxiv.org/pdf/1103.0398.pdf)
- [Karpathy's The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness)

### Videos and Online Courses

#### Deep Learning and NLP

Word embeddings, RNNs, LSTMs and CNNs for Natural Language Processing | [Back to Top](#contents)

- Udacity's [Intro to Artificial Intelligence](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) course which touches upon NLP as well
- Udacity's [Deep Learning](https://classroom.udacity.com/courses/ud730) using Tensorflow which covers a section on using deep learning for NLP tasks (covering Word2Vec, RNN's and LSTMs)
- Oxford's [Deep Natural Language Processing](https://github.com/oxford-cs-deepnlp-2017/lectures) has videos, lecture slides and reading material
- Stanford's [Deep Learning for Natural Language Processing (cs224-n)](https://web.stanford.edu/class/cs224n/) by Richard Socher and Christopher Manning
- Coursera's [Natural Language Processing](https://www.coursera.org/learn/language-processing) by National Research University Higher School of Economics
- Carnegie Mellon University's [Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/) by Language Technology Institute there

#### Classical NLP

Bayesian, statistics and Linguistics approaches for Natural Language Processing | [Back to Top](#contents)

- [Natural Language Processing by Prof. Mike Collins at Columbia](https://www.youtube.com/watch?v=mieV29RVpuQ&list=PL0ap34RKaADMjqjdSkWolD-W2VSCyRUQC)
- [Statistical Machine Translation](http://mt-class.org) - a Machine Translation course with great assignments and slides
- [NLTK with Python 3 for Natural Language Processing](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) by Harrison Kinsley(sentdex). Good tutorials with NLTK code implementation
- [Computational Linguistics I](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) by Jordan Boyd-Graber, Lectures from University of Maryland
- [Deep NLP Course](https://github.com/yandexdataschool/nlp_course) by Yandex Data School, covering important ideas from text embedding to machine translation including sequence modeling, language models and so on. 

### Books

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Prof. Dan Jurafsy
- [Text Mining in R](https://www.tidytextmining.com)
- [Natural Language Processing with Python](http://www.nltk.org/book/)

## Libraries

[Back to Top](#contents)

- <a id="node-js">**Node.js and Javascript** - Node.js Libaries for NLP</a> | [Back to Top](#contents)
  - [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library
  - [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS
  - [Retext](https://github.com/wooorm/retext) - Extensible system for analyzing and manipulating natural language
  - [NLP Compromise](https://github.com/nlp-compromise/nlp_compromise) - Natural Language processing in the browser
  - [Natural](https://github.com/NaturalNode/natural) - general natural language facilities for node
  - [Poplar](https://github.com/synyi/poplar) - A web-based annotation tool for natural language processing (NLP)

- <a id="python"> **Python** - Python NLP Libraries</a> | [Back to Top](#contents)

  - [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of [Natural Language Toolkit (NLTK)](http://www.nltk.org/) and [Pattern](https://github.com/clips/pattern), and plays nicely with both :+1:
  - [spaCy](https://github.com/spacy-io/spaCy) - Industrial strength NLP with Python and Cython :+1:
    - [textacy](https://github.com/chartbeat-labs/textacy) - Higher level NLP built on spaCy
  - [gensim](https://radimrehurek.com/gensim/index.html) - Python library to conduct unsupervised semantic modelling from plain text :+1:
  - [scattertext](https://github.com/JasonKessler/scattertext) - Python library to produce d3 visualizations of how language differs between corpora
  - [AllenNLP](https://github.com/allenai/allennlp) - An NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
  - [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP research toolkit designed to support rapid prototyping with better data loaders, word vector loaders, neural network layer representations, common NLP metrics such as BLEU
  - [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
  - [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](http://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
  - [jPTDP](https://github.com/datquocnguyen/jPTDP) - A toolkit for joint part-of-speech (POS) tagging and dependency parsing. jPTDP provides pre-trained models for 40+ languages.
  - [BigARTM](https://github.com/bigartm/bigartm) - a fast library for topic modelling
  - [Snips NLU](https://github.com/snipsco/snips-nlu) - A production ready library for intent parsing
  - [Chazutsu](https://github.com/chakki-works/chazutsu) - A library for downloading&parsing standard NLP research datasets
  - [Word Forms](https://github.com/gutfeeling/word_forms) - Word forms can accurately generate all possible forms of an English word
  - [Multilingual Latent Dirichlet Allocation (LDA)](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA) - A multilingual and extensible document clustering pipeline

- <a id="c++">**C++** - C++ Libraries</a> | [Back to Top](#contents)
  - [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
  - [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
  - [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data.
  - [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  - [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  - [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
  - [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](http://proycon.github.io/folia/)
  - [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
  - [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
  - [Mecab (Japanese)](http://taku910.github.io/mecab/)
  - [Moses](http://statmt.org/moses/)
  - [StarSpace](https://github.com/facebookresearch/StarSpace) - a library from Facebook for creating embeddings of word-level, paragraph-level, document-level and for text classification

- <a id="java">**Java** - Java NLP Libraries</a> | [Back to Top](#contents)
  - [Stanford NLP](http://nlp.stanford.edu/software/index.shtml)
  - [OpenNLP](http://opennlp.apache.org/)
  - [ClearNLP](https://github.com/clir/clearnlp)
  - [Word2vec in Java](http://deeplearning4j.org/word2vec.html)
  - [ReVerb](https://github.com/knowitall/reverb/) Web-Scale Open Information Extraction
  - [OpenRegex](https://github.com/knowitall/openregex) An efficient and flexible token-based regular expression language and engine.
  - [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - Core libraries developed in the U of Illinois' Cognitive Computation Group.
  - [MALLET](http://mallet.cs.umass.edu/) - MAchine Learning for LanguagE Toolkit - package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text.
  - [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - A robust POS tagging toolkit available  (in both Java & Python) together with pre-trained models for 40+ languages.

- <a id="scala">**Scala** - Scala NLP Libraries</a> | [Back to Top](#contents)
  - [Saul](https://github.com/CogComp/saul) - Library for developing NLP systems, including built in modules like SRL, POS, etc.
  - [ATR4S](https://github.com/ispras/atr4s) - Toolkit with state-of-the-art [automatic term recognition](https://en.wikipedia.org/wiki/Terminology_extraction) methods.
  - [tm](https://github.com/ispras/tm) - Implementation of topic modeling based on regularized multilingual [PLSA](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis).
  - [word2vec-scala](https://github.com/Refefer/word2vec-scala) - Scala interface to word2vec model; includes operations on vectors like word-distance and word-analogy.
  - [Epic](https://github.com/dlwh/epic) - Epic is a high performance statistical parser written in Scala, along with a framework for building complex structured prediction models.

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
  - [postagga](https://github.com/turbopape/postagga) - A library to parse natural language in Clojure and ClojureScript

- <a id="ruby">**Ruby**</a> | [Back to Top](#contents)
  - Kevin Dias's [A collection of Natural Language Processing (NLP) Ruby libraries, tools and software](https://github.com/diasks2/ruby-nlp)
  - [Practical Natural Language Processing done in Ruby](https://github.com/arbox/nlp-with-ruby)

- <a id="rust">**Rust**</a>
  - [whatlang](https://github.com/greyblake/whatlang-rs) — Natural language recognition library based on trigrams
  - [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) - A production ready library for intent parsing

### Services

NLP as API with higher level functionality such as NER, Topic tagging and so on | [Back to Top](#contents)

- [Wit-ai](https://github.com/wit-ai/wit) - Natural Language Interface for apps and devices
- [IBM Watson's Natural Language Understanding](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs) - API and Github demo 
- [Amazon Comprehend](https://aws.amazon.com/comprehend/) - NLP and ML suite covers most common tasks like NER, tagging, and sentiment analysis
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language/) - Syntax Analysis, NER, Sentiment Analysis, and Content tagging in atleast 9 languages include English and Chinese (Simplified and Traditional).
- [ParallelDots](https://www.paralleldots.com/text-analysis-apis) - High level Text Analysis API Service ranging from Sentiment Analysis to Intent Analysis
- [Microsoft Cognitive Service](https://www.microsoft.com/cognitive-services/en-us/text-analytics-api)
- [TextRazor](https://www.textrazor.com/)
- [Rosette](https://www.rosette.com/)

### Annotation Tools

- [GATE](https://gate.ac.uk/overview.html) - General Architecture and Text Engineering is 15+ years old, free and open source
- [Anafora](https://github.com/weitechen/anafora) is free and open source, web-based raw text annotation tool
- [brat](http://brat.nlplab.org/) - brat rapid annotation tool is an online environment for collaborative text annotation
- [tagotag](https://www.tagtog.net/), costs $
- [prodigy](https://prodi.gy/) is an annotation tool powered by active learning, costs $
- [LightTag](https://lighttag.io) - Hosted and managed text annotation tool for teams, costs $

## Techniques

### Text Embeddings

[Back to Top](#contents)

Text embeddings allow deep learning to be effective on smaller datasets. These are often first inputs to a deep learning archiectures and most popular way of transfer learning in NLP. Embeddings are simply vectors or a more generically, real valued representations of strings. Word embeddings are considered a great starting point for most deep NLP tasks.

The most popular names in word embeddings are word2vec by Google (Mikolov) and GloVe by Stanford (Pennington, Socher and Manning). fastText seems to be a fairly popular for multi-lingual sub-word embeddings. 

#### Word Embeddings

[Back to Top](#contents)

|Embedding |Paper| Organisation| gensim- Training Support |Blogs|
|---|---|---|---|---|
|word2vec|[Official Implementation](https://code.google.com/p/word2vec/), T.Mikolove et al. 2013. Distributed Representations of Words and Phrases and their Compositionality. [pdf](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |Google|Yes :heavy_check_mark:| Visual explanations by colah at [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/); gensim's [Making Sense of word2vec](http://rare-technologies.com/making-sense-of-word2vec) |
|GloVe|Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf)|Stanford|No :negative_squared_cross_mark:|[Morning Paper on GloVe](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/) by acoyler|
|fastText|[Official Implementation](https://github.com/facebookresearch/fastText), T. Mikolov et al. 2017. Enriching Word Vectors with Subword Information. [pdf](https://arxiv.org/abs/1607.04606)|Facebook|Yes :heavy_check_mark:|[Fasttext: Under the Hood](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)|

Notes for Beginners:

- Thumb Rule: **fastText >> GloVe > word2vec**
- You can find [pre-trained fasttext Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html) in several languages
- If you are interested in the logic and intuition behind word2vec and GloVe: [The Amazing Power of Word Vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/) and  introduce the topics well
- [arXiv: Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759), and [arXiv: FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651) were released as part of fasttext

#### Sentence and Language Model Based Word Embeddings

[Back to Top](#contents)

- _ElMo_ from [Deep Contextualized Word Represenations](https://arxiv.org/abs/1802.05365) - [PyTorch implmentation](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) - [TF Implementation](https://github.com/allenai/bilm-tf)
- _ULimFit_ aka [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder
- _InferSent_ from [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) by facebook
- _CoVe_ from [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
- _Pargraph vectors_ from [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf). See [doc2vec tutorial at gensim](http://rare-technologies.com/doc2vec-tutorial/)
- [sense2vec](http://arxiv.org/abs/1511.06388) - on word sense disambiguation
- [Skip Thought Vectors](http://arxiv.org/abs/1506.06726) - word representation method
- [Adaptive skip-gram](http://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties
- [Sequence to Sequence Learning](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - word vectors for machine translation

### Question Answering and Knowledge Extraction

[Back to Top](#contents)

- [Markov Logic Networks for Natural Language Question Answering](http://arxiv.org/pdf/1507.03045v1.pdf)
- [Template-Based Information Extraction without the Templates](http://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
- [Relation extraction with matrix factorization and universal schemas](http://www.anthology.aclweb.org/N/N13/N13-1008.pdf)
- [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](http://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
- [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340) - DeepMind paper
- [DrQA: Open Domain Question Answering](https://github.com/facebookresearch/DrQA) by facebook on Wikipedia data
- [Relation Extraction with Matrix Factorization and Universal Schemas](http://www.riedelcastro.org//publications/papers/riedel13relation.pdf)
- [Towards a Formal Distributional Semantics: Simulating Logical Calculi with Tensors](http://www.aclweb.org/anthology/S13-1001)
- [Presentation slides for MLN tutorial](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt)
- [Presentation slides for QA applications of MLNs](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf)
- [Presentation slides](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf)

## Datasets

[Back to Top](#contents)

- [nlp-datasets](https://github.com/niderhoff/nlp-datasets) great collection of nlp datasets

## Multilingual NLP Frameworks

[Back to Top](#contents)

- [UDPipe](https://github.com/ufal/udpipe) is a trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U files.  Primarily written in C++, offers a fast and reliable solution for multilingual NLP processing.
- [NLP-Cube](https://github.com/adobe/NLP-Cube) : Natural Language Processing Pipeline - Sentence Splitting, Tokenization, Lemmatization, Part-of-speech Tagging and Dependency Parsing. New platform, written in Python with Dynet 2.0. Offers standalone (CLI/Python bindings) and server functionality (REST API).

## NLP in Korean

[Back to Top](#contents)

### Libraries

- [KoNLPy](http://konlpy.org) - Python package for Korean natural language processing.
- [Mecab (Korean)](http://eunjeon.blogspot.com/) - C++ library for Korean NLP
- [KoalaNLP](https://nearbydelta.github.io/KoalaNLP/) - Scala library for Korean Natural Language Processing.
- [KoNLP](https://cran.r-project.org/web/packages/KoNLP/index.html) - R package for Korean Natural language processing

### Blogs and Tutorials

- [dsindex's blog](http://dsindex.github.io/)
- [Kangwon University's NLP course in Korean](http://cs.kangwon.ac.kr/~leeck/NLP/)

### Datasets

- [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus)- A corpus from the Korea Advanced Institute of Science and Technology in Korean.
- [Naver Sentiment Movie Corpus in Korean](https://github.com/e9t/nsmc/)
- [Chosun Ilbo archive](http://srchdb1.chosun.com/pdf/i_archive/) - dataset in Korean from one of the major newspapers in South Korea, the Chosun Ilbo.

## NLP in Arabic

[Back to Top](#contents)

### Libraries

- [goarabic](https://github.com/01walid/goarabic)-  Go package for Arabic text processing
- [jsastem](https://github.com/ejtaal/jsastem) - Javascript for Arabic stemming
- [PyArabic](https://pypi.python.org/pypi/PyArabic/0.4) - Python libraries for Arabic

### Datasets

- [Multidomain Datasets](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces) - Largest Available Multi-Domain Resources for Arabic Sentiment Analysis
- [LABR](https://github.com/mohamedadaly/labr) - LArge Arabic Book Reviews dataset
- [Arabic Stopwords](https://github.com/mohataher/arabic-stop-words) - A list of Arabic stopwords from various resources

## NLP in Chinese

[Back to Top](#contents)

### Libraries

- [jieba](https://github.com/fxsjy/jieba#jieba-1) - Python package for Words Segmentation Utilities in Chinese
- [SnowNLP](https://github.com/isnowfy/snownlp) - Python package for Chinese NLP
- [FudanNLP](https://github.com/FudanNLP/fnlp)- Java library for Chinese text processing

## NLP in German

- [German-NLP](https://github.com/adbar/German-NLP) - Curated list of open-access/open-source/off-the-shelf resources and tools developed with a particular focus on German
 
## NLP in Spanish

[Back to Top](#contents)

### Corpora

- [Columbian Political Speeches](https://github.com/dav009/LatinamericanTextResources)
- [Copenhagen Treebank](http://code.google.com/p/copenhagen-dependency-treebank/)
- [Reuters Corpora RCV2](http://trec.nist.gov/data/reuters/reuters.html)
- [Spanish Billion words corpus with Word2Vec embeddings](http://crscardellino.me/SBWCE/)

## NLP in Indic languages

[Back to Top](#contents)

### Hindi

### Corpora and Treebanks

- [Hindi Dependency Treebank](http://ltrc.iiit.ac.in/treebank_H2014/) - A multi-representational multi-layered treebank for Hindi and Urdu
- [Universal Dependencies Treebank in Hindi](http://universaldependencies.org/treebanks/hi/index.html)
  - [Parallel Universal Dependencies Treebank in Hindi](http://universaldependencies.org/treebanks/hi_pud/index.html) - A smaller part of the above-mentioned treebank.

## NLP in Thai

[Back to Top](#contents)

### Libraries

- [PyThaiNLP](https://github.com/wannaphongcom/pythainlp) - Thai NLP in Python Package
- [JTCC](https://github.com/wittawatj/jtcc)- A character cluster library in Java
- [CutKum](https://github.com/pucktada/cutkum) - Word segmentation with deep learning in TensorFlow
- [Thai Language Toolkit](https://pypi.python.org/pypi/tltk/) - Based on a paper by Wirote Aroonmanakun in 2002 with included dataset
- [SynThai](https://github.com/KenjiroAI/SynThai)- Word segmentation and POS tagging using deep learning in Python

### Corpora

- [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm) - A text corpus with 5 million words with word segmentation
- [Prime Minister 29](https://github.com/PyThaiNLP/lexicon-thai/tree/master/thai-corpus/Prime%20Minister%2029)- Dataset containing speeches of the current Prime Minister of Thailand

## NLP in Vietnamese

[Back to Top](#contents)

## NLP in Danish 

- [Named Entity Recognition for Danish](https://github.com/ITUnlp/daner)

### Libraries

- [underthesea](https://github.com/magizbox/underthesea) - Vietnamese NLP Toolkit
- [vn.vitk](https://github.com/phuonglh/vn.vitk) - A Vietnamese Text Processing Toolkit
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) - A Vietnamese natural language processing toolkit

### Corpora

- [Vietnamese treebank](https://vlsp.hpda.vn/demo/?page=resources&lang=en) - 10,000 sentences for the constituency parsing task
- [BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf) - a Vietnamese Dependency Treebank
- [UD_Vietnamese](https://github.com/UniversalDependencies/UD_Vietnamese-VTB) - Vietnamese Universal Dependency Treebank
- [VIVOS](https://ailab.hcmus.edu.vn/vivos/) - a free Vietnamese speech corpus consisting of 15 hours of recording speech by AILab
- [VNTQcorpus(big).txt](http://viet.jnlp.org/download-du-lieu-tu-vung-corpus) - 1.75 million sentences in news

### Other Languages 

- Russian: [pymorphy2](https://github.com/kmike/pymorphy2) - a good pos-tagger for Russian
- Asian Languages: Thai, Lao, Chinese, Japanese, and Korean [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html) implementation in ElasticSearch
- Ancient Languages: [CLTK](https://github.com/cltk/cltk): The Classical Language Toolkit is a Python library and collection of texts for doing NLP in ancient languages
- Dutch: [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
- Hebrew: [NLPH_Resources](https://github.com/NLPH/NLPH_Resources) - A collection of papers, corpora and linguistic resources for NLP in Hebrew

## Credits

Awesome NLP was seeded with curated content from the lot of repositories, some of which are listed below | [Back to Top](#contents)

- [ai-reading-list](https://github.com/m0nologuer/AI-reading-list)
- [nlp-reading-group](https://github.com/clulab/nlp-reading-group/wiki/Fall-2015-Reading-Schedule/_edit)
- [awesome-spanish-nlp](https://github.com/dav009/awesome-spanish-nlp)
- [jjangsangy's awesome-nlp](https://gist.github.com/jjangsangy/8759f163bc3558779c46)
- [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning/edit/master/README.md)
- [DL4NLP](https://github.com/andrewt3000/DL4NLP)

[Back to Top](#contents)
