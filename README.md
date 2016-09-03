# awesome-nlp [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of resources dedicated to Natural Language Processing
>
> Maintainers - [Keon Kim](https://github.com/keonkim), [Martin Park](https://github.com/outpark)

*Please read the [contribution guidelines](contributing.md) before contributing.*

Please feel free to [pull requests](https://github.com/keonkim/awesome-nlp/pulls), or email Martin Park (sp3005@nyu.edu)/Keon Kim (keon.kim@nyu.edu) to add links.


## Table of Contents

 - [Tutorials and Courses](#tutorials-and-courses)
   - [videos](#videos)
 - [Deep Learning for NLP](#deep-learning-for-nlp)
 - [Packages](#packages)
   - [Implemendations](#implementations)
   - [Libraries](#libraries)
     - [Node.js](#user-content-node-js)
     - [Python](#user-content-python)
     - [C++](#user-content-c++)
     - [Java](#user-content-java)
     - [Clojure](#user-content-clojure)
     - [Ruby](#user-content-ruby)
   - [Services](#services)
 - [Articles](#articles)
   - [Review Articles](#review-articles)
   - [Word Vectors](#word-vectors)
   - [Thought Vectors](#thought-vectors)
   - [Machine Translation](#machine-translation)
   - [General Natural Language Processing](#general-natural-langauge-processing)
   - [Named Entity Recognition](#name-entity-recognition)
   - [Single Exchange Dialogs](#single-exchange-dialogs)
   - [Memory and Attention Models](#memory-and-attention-models)
   - [General Natural Language Processing](#general-natural-language-processing)
   - [Named Entity Recognition](#named-entity-recognition)
   - [Neural Network](#neural-network)
   - [Supplementary Materials](#supplementary-materials)
 - [Blogs](#blogs)
 - [Credits](#credits)


## Tutorials and Courses

* Tensor Flow Tutorial on [Seq2Seq](https://www.tensorflow.org/tutorials/seq2seq/index.html) Models
* Natural Language Understanding with Distributed Representation [Lecture Note](https://github.com/nyu-dl/NLP_DL_Lecture_Note) by Cho
* [Michael Collins](http://www.cs.columbia.edu/~mcollins/) - one of the best NLP teachers. Check out the material on the courses he is teaching.

### videos

* [Intro to Natural Language Processing](https://www.coursera.org/learn/natural-language-processing) on Coursera by U of Michigan
* [Intro to Artificial Intelligence](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) course on Udacity which also covers NLP
* [Deep Learning for Natural Language Processing (2015 classes)](https://www.youtube.com/playlist?list=PLmImxx8Char8dxWB9LRqdpCTmewaml96q) by Richard Socher
* [Deep Learning for Natural Language Processing (2016 classes)](https://www.youtube.com/playlist?list=PLmImxx8Char9Ig0ZHSyTqGsdhb9weEGam) by Richard Socher. Updated to make use of Tensorflow. Note that there are some lectures missing (lecture 9, and lectures 12 onwards). 
* [Natural Language Processing](https://www.coursera.org/learn/nlangp) - course on Coursera that was only done in 2013. The videos are not available at the moment. Also Mike Collins is a great professor and his notes and lectures are very good. 
* [Statistical Machine Translation](http://mt-class.org) - a Machine Translation course with great assignments and slides. 
* [Natural Language Processing SFU](http://www.cs.sfu.ca/~anoop/teaching/CMPT-413-Spring-2014/) - course by [Prof Anoop Sarkar](https://www.cs.sfu.ca/~anoop/) on Natural Language Processing. Good notes and some good lectures on youtube about HMM. 
* [Udacity Deep Learning](https://classroom.udacity.com/courses/ud730) Deep Learning course on Udacity (using Tensorflow) which covers a section on using deep learning for NLP tasks (covering Word2Vec, RNN's and LSTMs).
* [NLTK with Python 3 for Natural Language Processing](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) by Harrison Kinsley(sentdex). Good tutorials with NLTK code implementation.

## Deep Learning for NLP 

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
Class by [Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). 2016 content was updated to make use of Tensorflow. Lecture slides and reading materials for 2016 class [here](http://cs224d.stanford.edu/syllabus.html). Videos for 2016 class [here](https://www.youtube.com/playlist?list=PLmImxx8Char9Ig0ZHSyTqGsdhb9weEGam). Note that there are some lecture videos missing for 2016 (lecture 9, and lectures 12 onwards). All videos for 2015 class [here](https://www.youtube.com/playlist?list=PLmImxx8Char8dxWB9LRqdpCTmewaml96q)

[Udacity Deep Learning](https://classroom.udacity.com/courses/ud730)
Deep Learning course on Udacity (using Tensorflow) which covers a section on using deep learning for NLP tasks. This section covers how to implement Word2Vec, RNN's and LSTMs.

[A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)  
Yoav Goldberg. October 2015. No new info, 75 page summary of state of the art.  


## Packages

### Implementations
* [Pre-trained word embeddings for WSJ corpus](https://github.com/ai-ku/wvec) by Koc AI-Lab
* [Word2vec](https://code.google.com/archive/p/word2vec) by Mikolov
* [HLBL language model](http://metaoptimize.com/projects/wordreprs/) by Turian
* [Real-valued vector "embeddings"](http://www.cis.upenn.edu/~ungar/eigenwords/) by Dhillon
* [Improving Word Representations Via Global Context And Multiple Word Prototypes](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes) by Huang
* [Dependency based word embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [Global Vectors for Word Representations](http://nlp.stanford.edu/projects/glove/)

### Libraries
* [TwitIE: An Open-Source Information Extraction Pipeline for Microblog Text](http://www.anthology.aclweb.org/R/R13/R13-1011.pdf)

* <a id="node-js">**Node.js and Javascript** - Node.js Libaries for NLP</a>
  * [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library
  * [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS
  * [Retext](https://github.com/wooorm/retext) - Extensible system for analyzing and manipulating natural language
  * [NLP Compromise](https://github.com/nlp-compromise/nlp_compromise) - Natural Language processing in the browser
  * [Natural](https://github.com/NaturalNode/natural) - general natural language facilities for node

* <a id="python">**Python** - Python NLP Libraries</a>
  * [Scikit-learn: Machine learning in Python](http://arxiv.org/pdf/1201.0490.pdf)
  * [Natural Language Toolkit (NLTK)](http://www.nltk.org/)
  * [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
  * [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
  * [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora.
  * [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
  * [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
  * [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
  * [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
  * [BLLIP Parser](https://pypi.python.org/pypi/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  * [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](http://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
  * [python-ucto](https://github.com/proycon/python-ucto) - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages)
  * [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
  * [python-zpar](https://github.com/EducationalTestingService/python-zpar) - Python bindings for [ZPar](https://github.com/frcchang/zpar), a statistical part-of-speech-tagger, constiuency parser, and dependency parser for English.
  * [colibri-core](https://github.com/proycon/colibri-core) - Python binding to C++ library for extracting and working with with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  * [spaCy](https://github.com/spacy-io/spaCy) - Industrial strength NLP with Python and Cython.
  * [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.

* <a id="c++">**C++** - C++ Libraries</a>
  * [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
  * [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
  * [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data.
  * [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  * [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  * [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
  * [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](http://proycon.github.io/folia/)
  * [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
  * [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
  * [Mecab (Japanese)](http://taku910.github.io/mecab/)
  * [Mecab (Korean)](http://eunjeon.blogspot.com/)
  * [Moses](http://statmt.org/moses/)

* <a id="java">**Java** - Java NLP Libraries</a>
  * [Stanford NLP](http://nlp.stanford.edu/software/index.shtml)
  * [OpenNLP](http://opennlp.apache.org/)
  * [ClearNLP](https://github.com/clir/clearnlp)
  * [Word2vec in Java](http://deeplearning4j.org/word2vec.html)
  * [ReVerb](https://github.com/knowitall/reverb/) Web-Scale Open Information Extraction
  * [OpenRegex](https://github.com/knowitall/openregex) An efficient and flexible token-based regular expression language and engine.  
  * [CogcompNLP](https://github.com/IllinoisCogComp/illinois-cogcomp-nlp) - Core libraries developed in the U of Illinois' Cognitive Computation Group. 
  
* <a id="scala">**Scala** - Scala NLP Libraries</a>
  * [Saul](https://github.com/IllinoisCogComp/saul) - Library for developing NLP systems, including built in modules like SRL, POS, etc. 

* <a id="clojure">**Clojure**</a>
  * [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
  * [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript

* <a id="ruby">**Ruby**</a>
  * Kevin Dias's [A collection of Natural Language Processing (NLP) Ruby libraries, tools and software](https://github.com/diasks2/ruby-nlp)
  
### Services
* [Wit-ai](https://github.com/wit-ai/wit) - Natural Language Interface for apps and devices.

## Articles

### Review Articles
* [Deep Learning for Web Search and Natural Language Processing](http://research.microsoft.com:8082/en-us/um/people/jfgao/paper/2015/wsdm2015.v3.pdf)
* [Probabilistic topic models](https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf)
* [Natural language processing: an introduction](http://jamia.oxfordjournals.org/content/18/5/544.short)
* [A unified architecture for natural language processing: Deep neural networks with multitask learning](http://arxiv.org/pdf/1201.0490.pdf)
* [A Critical Review of Recurrent Neural Networksfor Sequence Learning](http://arxiv.org/pdf/1506.00019v1.pdf)
* [Deep parsing in Watson](http://nlp.cs.rpi.edu/course/spring14/deepparsing.pdf)
* [Online named entity recognition method for microtexts in social networking services: A case study of twitter](http://arxiv.org/pdf/1301.2857.pdf)


### Word Vectors
Resources about word vectors, aka word embeddings, and distributed representations for words.  
Word vectors are numeric representations of words that are often used as input to deep learning systems. This process is sometimes called pretraining.  

[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013.  
Generate word and phrase vectors.  Performs well on word similarity and analogy task and includes [Word2Vec source code](https://code.google.com/p/word2vec/)  Subsamples frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)  
[Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)  

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)  
Chris Olah (2014)  Blog post explaining word2vec.  

[GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)  
Pennington, Socher, Manning. 2014. Creates word vectors and relates word2vec to matrix factorizations.  [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) by [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)  
[Glove source code and training data](http://nlp.stanford.edu/projects/glove/)

* [word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - on creating vectors to represent language, useful for RNN inputs
* [sense2vec](http://arxiv.org/abs/1511.06388) - on word sense disambiguation
* [Infinite Dimensional Word Embeddings](http://arxiv.org/abs/1511.05392) - new
* [Skip Thought Vectors](http://arxiv.org/abs/1506.06726) - word representation method
* [Adaptive skip-gram](http://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties

### Thought Vectors
Thought vectors are numeric representations for sentences, paragraphs, and documents.  The following papers are listed in order of date published, each one replaces the last as the state of the art in sentiment analysis.  

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)  
Socher et al. 2013.  Introduces Recursive Neural Tensor Network.  Uses a parse tree.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree.  
Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)

[Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)  
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks. Uses a parse tree.

[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Tai et al. 2015  Introduces Tree LSTM. Uses a parse tree.

[Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)  
Dai, Le 2015 "With pretraining, we are able to train long short term memory recurrent networks up to a few hundred
timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."  
### Machine Translation
[Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf)
Bahdanau, Cho 2014.  "comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation."  Implements attention mechanism.  
[English to French Demo](http://104.131.78.120/)  

[Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf)  
Sutskever, Vinyals, Le 2014.  ([nips presentation](http://research.microsoft.com/apps/video/?id=239083)). Uses LSTM RNNs to generate translations. " Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8"  
[seq2seq tutorial](http://tensorflow.org/tutorials/seq2seq/index.html) in 

* [Cross-lingual Pseudo-Projected Expectation Regularization for Weakly Supervised Learning](http://arxiv.org/pdf/1310.1597v1.pdf)
* [Generating Chinese Named Entity Data from a Parallel Corpus](http://www.mt-archive.info/IJCNLP-2011-Fu.pdf)
* [IXA pipeline: Efficient and Ready to Use Multilingual NLP tools](http://www.lrec-conf.org/proceedings/lrec2014/pdf/775_Paper.pdf)


### Single Exchange Dialogs
[A Neural Network Approach toContext-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni 2015.  Generates responses to tweets.   
Uses [Recurrent Neural Network Language Model (RLM) architecture
of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)  source code: [RNNLM Toolkit](http://www.rnnlm.org/)

[Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)  
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.  

[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses. Uses [seq2seq framework](http://tensorflow.org/tutorials/seq2seq/index.html).  Seq2Seq was originally designed for machine transation and it "translates" a single sentence, up to around 79 words, to a single sentence response, and has no memory of previous dialog exchanges.  Used in Google [Smart Reply feature for Inbox](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)  

### Memory and Attention Models (from [DL4NLP](https://github.com/andrewt3000/DL4NLP))
[Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)  

[Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf) Weston et. al 2014, and 
[End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf) Sukhbaatar et. al 2015.  
Memory networks are implemented in [MemNN](https://github.com/facebook/MemNN).  Attempts to solve task of reason attention and memory.  
[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks like single factoid, yes/no etc. Extends memory networks.  
[Evaluating prerequisite qualities for learning end to end dialog systems](http://arxiv.org/pdf/1511.06931.pdf)  
Dodge et. al 2015. Tests Memory Networks on 4 tasks including reddit dialog task.  
See [Jason Weston lecture on MemNN](https://www.youtube.com/watch?v=Xumy3Yjq4zk)  
  
[Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)  
Graves et al. 2014.  

[Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf)  
Joulin, Mikolov 2015. [Stack RNN source code](https://github.com/facebook/Stack-RNN) and [blog post](https://research.facebook.com/blog/1642778845966521/inferring-algorithmic-patterns-with-stack/)  

### General Natural Language Processing
* [Neural autocoder for paragraphs and documents](http://arxiv.org/abs/1506.01057) - LSTM representation
* [LSTM over tree structures](http://arxiv.org/abs/1503.04881)
* [Sequence to Sequence Learning](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - word vectors for machine translation
* [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340) - DeepMind paper
* [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)
* [Improving distributional similarity with lessons learned from word embeddings](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/viewFile/570/124)
* [Low-Dimensional Embeddings of Logic](http://www.aclweb.org/anthology/W/W14/W14-2409.pdf)
* Tutorial on Markov Logic Networks ([based on this paper](http://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf))
* [Markov Logic Networks for Natural Language Question Answering](http://arxiv.org/pdf/1507.03045v1.pdf)
* [Distant Supervision for Cancer Pathway Extraction From Text](http://research.microsoft.com/en-us/um/people/hoifung/papers/psb15.pdf)
* [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](http://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
* [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
* [Template-Based Information Extraction without the Templates](http://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
* [Retrofitting word vectors to semantic lexicons](http://www.cs.cmu.edu/~mfaruqui/papers/naacl15-retrofitting.pdf)
* [Unsupervised Learning of the Morphology of a Natural Language](http://www.mitpressjournals.org/doi/pdfplus/10.1162/089120101750300490)
* [Natural Language Processing (Almost) from Scratch](http://arxiv.org/pdf/1103.0398.pdf)
* [Computational Grounded Cognition: a new alliance between grounded cognition and computational modelling](http://journal.frontiersin.org/article/10.3389/fpsyg.2012.00612/full)
* [Learning the Structure of Biomedical Relation Extractions](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004216)
* [Relation extraction with matrix factorization and universal schemas](http://www.anthology.aclweb.org/N/N13/N13-1008.pdf)

### Named Entity Recognition
* [A survey of named entity recognition and classification](http://nlp.cs.nyu.edu/sekine/papers/li07.pdf)
* [Benchmarking the extraction and disambiguation of named entities on the semantic web](http://www.lrec-conf.org/proceedings/lrec2014/pdf/176_Paper.pdf)
* [Knowledge base population: Successful approaches and challenges](http://www.aclweb.org/anthology/P11-1115)
* [SpeedRead: A fast named entity recognition Pipeline](http://arxiv.org/pdf/1301.2857.pdf)

### Neural Network
* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
* [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
* [Slides from Google Talk](http://www.fit.vutbr.cz/~imikolov/rnnlm/google.pdf)

### Supplementary Materials
* [Word2Vec](https://github.com/clulab/nlp-reading-group/wiki/Word2Vec-Resources)
* [Relation Extraction with Matrix Factorization and Universal Schemas](http://www.riedelcastro.org//publications/papers/riedel13relation.pdf)
* [Towards a Formal Distributional Semantics: Simulating Logical Calculi with Tensors](http://www.aclweb.org/anthology/S13-1001)
* [Presentation slides for MLN tutorial](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt)
* [Presentation slides for QA applications of MLNs](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf)
* [Presentation slides](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf)
* [Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations](https://homes.cs.washington.edu/~clzhang/paper/acl2011.pdf)


## Blogs
* Blog Post on [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* Blog Post on [NLP Tutorial](http://www.vikparuchuri.com/blog/natural-language-processing-tutorial/)
* [Natural Language Processing Blog](http://nlpers.blogspot.ch/) by Hal Daumé III
* [Machine Learning Blog](https://bmcfee.github.io/#home) by Brian McFee


## Credits
part of the lists are from 
* [ai-reading-list](https://github.com/m0nologuer/AI-reading-list) 
* [nlp-reading-group](https://github.com/clulab/nlp-reading-group/wiki/Fall-2015-Reading-Schedule/_edit)
* [awesome-spanish-nlp](https://github.com/dav009/awesome-spanish-nlp)
* [jjangsangy's awesome-nlp](https://gist.github.com/jjangsangy/8759f163bc3558779c46)
* [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning/edit/master/README.md)
* [DL4NLP](https://github.com/andrewt3000/DL4NLP)
