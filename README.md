# awesome-nlp
A curated list of resources dedicated to Natural Language Processing


Maintainers - [Keon Kim](http://github.com/keonkim), [Martin Park](https://github.com/outpark)

## Contributing
Please feel free to [pull requests](https://github.com/keonkim/awesome-nlp/pulls), email Keon Kim (keon.kim@nyu.edu) to add links.


## Table of Contents

 - [Tutorials and Courses](#tutorials-and-courses)
   - [videos](#videos)
 - [Codes](#codes)
   - [Implemendations](#implementations)
   - [Libraries](#libraries)
     - [Node.js](#user-content-node-js)
     - [Python](#user-content-python)
     - [C++](#user-content-c++)
     - [Java](#user-content-java)
     - [Clojure](#user-content-clojure)
 - [Articles](#articles)
   - [Review Articles](#review-articles)
   - [Word Vectors](#word-vectors)
   - [General Natural Language Processing](#general-natural-langauge-processing)
   - [Named Entity Recognition](#name-entity-recognition)
   - [Machine Translation](#machine-translation)
   - [Neural Network](#neural-network)
   - [Supplementary Materials](#supplementary-materials)
 - [Blogs](#blogs)
 - [Multilingual](#multilingual)
   - [Spanish](#spanish)
 - [Credits](#credits)


## Tutorials and Courses

* Tensor Flow Tutorial on [Seq2Seq](http://www.tensorflow.org/tutorials/seq2seq/index.html) Models

### videos
* [Stanford's Coursera Course](https://www.coursera.org/course/nlp) on NLP from basics
* [Intro to Natural Language Processing](https://www.coursera.org/course/nlpintro) on Coursera by U of Michigan
* [Intro to Artificial Intelligence](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) course on Udacity which also covers NLP
* [Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/) by Richard Socher

## Codes

### Implementations
* [Pre-trained word embeddings for WSJ corpus](https://github.com/ai-ku/wvec) by Koc AI-Lab
* [Word2vec](https://code.google.com/p/word2vec/) by Mikolov
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
  * [NLP Compromise](https://github.com/spencermountain/nlp_compromise) - Natural Language processing in the browser
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
  * [spaCy](https://github.com/honnibal/spaCy/) - Industrial strength NLP with Python and Cython.
  * [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.

* <a id="c++">**C++** - C++ Libraries</a>
  * [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
  * [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
  * [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data.
  * [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  * [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  * [ucto](https://github.com/proycon/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
  * [libfolia](https://github.com/proycon/libfolia) - C++ library for the [FoLiA format](http://proycon.github.io/folia/)
  * [frog](https://github.com/proycon/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
  * [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
  * [Mecab (Japanese)](http://taku910.github.io/mecab/)
  * [Mecab (Korean)](http://eunjeon.blogspot.com/)

* <a id="java">**Java** - Java NLP Libraries</a>
  * [Stanford NLP](http://nlp.stanford.edu/software/index.shtml)
  * [OpenNLP](http://opennlp.apache.org/)
  * [ClearNLP](https://github.com/clir/clearnlp)
  * [Word2vec in Java](http://deeplearning4j.org/word2vec.html)
  * [ReVerb](https://github.com/knowitall/reverb/) Web-Scale Open Information Extraction
  * [OpenRegex](https://github.com/knowitall/openregex) An efficient and flexible token-based regular expression language and engine.  
  
* <a id="clojure">**Clojure**</a>
  * [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
  * [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript

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
* [word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - on creating vectors to represent language, useful for RNN inputs
* [sense2vec](http://arxiv.org/abs/1511.06388) - on word sense disambiguation
* [Infinite Dimensional Word Embeddings](http://arxiv.org/abs/1511.05392) - new
* [Skip Thought Vectors](http://arxiv.org/abs/1506.06726) - word representation method
* [Adaptive skip-gram](http://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties

### General Natural Language Processing
* [Neural autocoder for paragraphs and documents](http://arxiv.org/abs/1506.01057) - LTSM representation
* [LTSM over tree structures](http://arxiv.org/abs/1503.04881)
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

### Machine Translation
* [Cross-lingual Pseudo-Projected Expectation Regularization for Weakly Supervised Learning](http://arxiv.org/pdf/1310.1597v1.pdf)
* [Generating Chinese Named Entity Data from a Parallel Corpus](http://www.mt-archive.info/IJCNLP-2011-Fu.pdf)
* [IXA pipeline: Efficient and Ready to Use Multilingual NLP tools](http://www.lrec-conf.org/proceedings/lrec2014/pdf/775_Paper.pdf)

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

## Multilingual

### Spanish


- POS TAGGERS
   - [TreeTagger - POSTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
   - [Stanford - POSTagger](http://nlp.stanford.edu/software/tagger.shtml)
   - [Freeling](http://nlp.lsi.upc.edu/freeling/)
   - [ixa-pipe-pos](https://github.com/ixa-ehu/ixa-pipe-pos)
   - [Ruby Snowball Implementation](https://github.com/MaG21/estem)
   - [Spaguetti POSTagger(Based on NLTK +  CESS corpus](https://code.google.com/p/spaghetti-tagger/)
- NER
   - [OpenNLP - Person/Place/Organization models](http://opennlp.sourceforge.net/models-1.5/)
   - [DBPedia Spotlight](https://github.com/dbpedia-spotlight/dbpedia-spotlight/)
   - [CitiusTagger - Spanish NER and  POSTagger](http://gramatica.usc.es/pln/tools/CitiusTools.html)
- ETC
   - [Word2Vec vectors for Wikipedia Spanish Articles](https://github.com/idio/wiki2vec)
   - [DBpedia Spanish Entities Titles](http://data.dws.informatik.uni-mannheim.de/dbpedia/2014/es/labels_es.nt.bz2)
   - [DBpedia Spanish Abstracts](http://data.dws.informatik.uni-mannheim.de/dbpedia/2014/es/short_abstracts_es.nt.bz2)
   - [Conshuga - Galician Verb conjugator](http://gramatica.usc.es/pln/tools/conjugador/download.html)


## Credits
part of the lists are from 
* [ai-reading-list](https://github.com/m0nologuer/AI-reading-list) 
* [nlp-reading-group](https://github.com/clulab/nlp-reading-group/wiki/Fall-2015-Reading-Schedule/_edit)
* [awesome-spanish-nlp](https://github.com/dav009/awesome-spanish-nlp)
* [jjangsangy's awesome-nlp](https://gist.github.com/jjangsangy/8759f163bc3558779c46)
* [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning/edit/master/README.md)
