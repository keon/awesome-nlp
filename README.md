# awesome-nlp 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of resources dedicated to Natural Language Processing
>
> Maintainers - [Keon](https://github.com/keon), [Martin](https://github.com/outpark), [Nirant](https://github.com/NirantK), [Dhruv](https://github.com/the-ethan-hunt)

*Please read the [contribution guidelines](contributing.md) before contributing.*

Please feel free to create [pull requests](https://github.com/keonkim/awesome-nlp/pulls).


## Contents

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
 - [Techniques](#techniques)
   - [Text Embeddings](#text-embeddings)
   - [Thought Vectors](#thought-vectors)
   - [Machine Translation](#machine-translation)
   - [Dialogs and Conversational](#dialogs-and-conversational)
   - [Memory and Attention Models](#memory-and-attention-models)
   - [Named Entity Recognition](#named-entity-recognition)
   - [Natural Language Understanding](#natural-language-understanding)
   - [Question Answering and Knowledge Extraction](#question-answering-and-knowledge-extraction)
   - [Text Summarization](#text-summarization)
   - [Text Classification](#text-classification)
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


## Tutorials
[Back to Top](#contents)

### Reading Content

General Machine Learning
* [AI Playbook](http://aiplaybook.a16z.com/) is a brief set of pieces to introduce machine learning and other advancements to technical as well as non-technical audience. Written by the amazing people over at [a16z - Andreessen Horowitz](https://a16z.com/) this is a great link to forward to your managers or content for your presentations 
* [Machine Learning Blog](https://bmcfee.github.io/#home) by Brian McFee
* [Ruder's Blog](http://ruder.io/#open) by [Sebastian Ruder](https://twitter.com/seb_ruder) for commentary on the best of NLP Research

Introductions and Guides to NLP
* Ultimate Guide to [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [Introduction to NLP at Hackernoon](https://hackernoon.com/learning-ai-if-you-suck-at-math-p7-the-magic-of-natural-language-processing-f3819a689386) is for people who suck at math - in their own words
* [NLP Tutorial by Vik Paruchari](http://www.vikparuchuri.com/blog/natural-language-processing-tutorial/)
* [Natural Language Processing: An Introduction](http://jamia.oxfordjournals.org/content/18/5/544.short) by Oxford
* [Deep Learning for NLP with Pytorch](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
* [Hands-On NLTK Tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) - The hands-on NLTK tutorial in the form of Jupyter notebooks


Blogs and Newsletters
* [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* [Natural Language Processing Blog](http://nlpers.blogspot.ch/) by Hal Daumé III
* [Tutorials by Radim Řehůřek](https://radimrehurek.com/gensim/tutorial.html) on using Python and [gensim](https://radimrehurek.com/gensim/index.html) to process language corpora 
* [arXiv: Natural Language Processing (Almost) from Scratch](http://arxiv.org/pdf/1103.0398.pdf)
* [Karpathy's The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
* [Sebastian Ruder](http://ruder.io/#open)'s blog is focused on NLP Research

### Videos and Online Courses

#### Deep Learning and NLP
Word embeddings, RNNs, LSTMs and CNNs for Natural Language Processing | [Back to Top](#contents)
* [Udacity's Intro to Artificial Intelligence](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) course which touches upon NLP as well
* [Udacity's Deep Learning](https://classroom.udacity.com/courses/ud730) using Tensorflow which covers a section on using deep learning for NLP tasks (covering Word2Vec, RNN's and LSTMs)
* [Deep Natural Language Processing at Oxford](https://github.com/oxford-cs-deepnlp-2017/lectures) has videos, lecture slides and reading material
* [Deep Learning for Natural Language Processing (cs224*n*)](web.stanford.edu/class/cs224n/) by Richard Socher and Christopher Manning at Stanford. Includes Videos, Assignments, Syllabus and other detailed material
[Lecture Slides and Reading Material here](http://web.stanford.edu/class/cs224n/)
* [Coursera's Natural Language Processing](https://www.coursera.org/learn/language-processing) by National Research University Higher School of Economics

#### Classical NLP
Bayesian, statistics and Linguistics approaches for Natural Language Processing | [Back to Top](#contents)
* [Natural Language Processing by Prof. Mike Collins at Columbia](https://www.youtube.com/watch?v=mieV29RVpuQ&list=PL0ap34RKaADMjqjdSkWolD-W2VSCyRUQC)
* [Statistical Machine Translation](http://mt-class.org) - a Machine Translation course with great assignments and slides
* [NLTK with Python 3 for Natural Language Processing](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) by Harrison Kinsley(sentdex). Good tutorials with NLTK code implementation
* [Computational Linguistics I](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) by Jordan Boyd-Graber, Lectures from University of Maryland

### Books

* [Text Mining in R](https://www.tidytextmining.com)
* [Natural Language Processing with Python](http://www.nltk.org/book/)
* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)

## Libraries

[Back to Top](#contents)

* <a id="node-js">**Node.js and Javascript** - Node.js Libaries for NLP</a> | [Back to Top](#contents)
  * [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library
  * [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS
  * [Retext](https://github.com/wooorm/retext) - Extensible system for analyzing and manipulating natural language
  * [NLP Compromise](https://github.com/nlp-compromise/nlp_compromise) - Natural Language processing in the browser
  * [Natural](https://github.com/NaturalNode/natural) - general natural language facilities for node

* <a id="python">**Python** - Python NLP Libraries</a> | [Back to Top](#contents)

   * [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of [Natural Language Toolkit (NLTK)](http://www.nltk.org/) and [Pattern](https://github.com/clips/pattern), and plays nicely with both :+1:
   * [spaCy](https://github.com/spacy-io/spaCy) - Industrial strength NLP with Python and Cython :+1:
     * [textacy](https://github.com/chartbeat-labs/textacy) - Higher level NLP built on spaCy
   * [gensim](https://radimrehurek.com/gensim/index.html) - Python library to conduct unsupervised semantic modelling from plain text :+1:
   * [scattertext](https://github.com/JasonKessler/scattertext) - Python library to produce d3 visualizations of how language differs between corpora
   * [AllenNLP](https://github.com/allenai/allennlp) - An NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
   * [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP research toolkit designed to support rapid prototyping with better data loaders, word vector loaders, neural network layer representations, common NLP metrics such as BLEU
   * [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
   * [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](http://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
   * [jPTDP](https://github.com/datquocnguyen/jPTDP) - A toolkit for joint part-of-speech (POS) tagging and dependency parsing. jPTDP provides pre-trained models for 40+ languages.
   * [BigARTM](https://github.com/bigartm/bigartm) - a fast library for topic modelling
   * [Snips NLU](https://github.com/snipsco/snips-nlu) - A production ready library for intent parsing
   

* <a id="c++">**C++** - C++ Libraries</a> | [Back to Top](#contents)
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
  * [Moses](http://statmt.org/moses/)
  * [StarSpace](https://github.com/facebookresearch/StarSpace) - a library from Facebook for creating embeddings of word-level, paragraph-level, document-level and for text classification

* <a id="java">**Java** - Java NLP Libraries</a> | [Back to Top](#contents)
  * [Stanford NLP](http://nlp.stanford.edu/software/index.shtml)
  * [OpenNLP](http://opennlp.apache.org/)
  * [ClearNLP](https://github.com/clir/clearnlp)
  * [Word2vec in Java](http://deeplearning4j.org/word2vec.html)
  * [ReVerb](https://github.com/knowitall/reverb/) Web-Scale Open Information Extraction
  * [OpenRegex](https://github.com/knowitall/openregex) An efficient and flexible token-based regular expression language and engine.
  * [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - Core libraries developed in the U of Illinois' Cognitive Computation Group.
  * [MALLET](http://mallet.cs.umass.edu/) - MAchine Learning for LanguagE Toolkit - package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text.
  * [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - A robust POS tagging toolkit available  (in both Java & Python) together with pre-trained models for 40+ languages.

* <a id="scala">**Scala** - Scala NLP Libraries</a> | [Back to Top](#contents)
  * [Saul](https://github.com/CogComp/saul) - Library for developing NLP systems, including built in modules like SRL, POS, etc.
  * [ATR4S](https://github.com/ispras/atr4s) - Toolkit with state-of-the-art [automatic term recognition](https://en.wikipedia.org/wiki/Terminology_extraction) methods.
  * [tm](https://github.com/ispras/tm) - Implementation of topic modeling based on regularized multilingual [PLSA](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis).
  * [word2vec-scala](https://github.com/Refefer/word2vec-scala) - Scala interface to word2vec model; includes operations on vectors like word-distance and word-analogy.
  * [Epic](https://github.com/dlwh/epic) - Epic is a high performance statistical parser written in Scala, along with a framework for building complex structured prediction models.

* <a id="R">**R** - R NLP Libraries</a> | [Back to Top](#contents)
  * [text2vec](https://github.com/dselivanov/text2vec) - Fast vectorization, topic modeling, distances and GloVe word embeddings in R.
  * [wordVectors](https://github.com/bmschmidt/wordVectors) - An R package for creating and exploring word2vec and other word embedding models
  * [RMallet](https://github.com/mimno/RMallet) - R package to interface with the Java machine learning tool MALLET
  * [dfr-browser](https://github.com/agoldst/dfr-browser) - Creates d3 visualizations for browsing topic models of text in a web browser.
  * [dfrtopics](https://github.com/agoldst/dfrtopics) - R package for exploring topic models of text.
  * [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - Sentiment Classification using Word Sense Disambiguation and WordNet Reader
  * [jProcessing](https://github.com/kevincobain2000/jProcessing) - Japanese Natural Langauge Processing Libraries, with Japanese sentiment classification

* <a id="clojure">**Clojure**</a> | [Back to Top](#contents)
  * [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
  * [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript
  * [postagga](https://github.com/turbopape/postagga) - A library to parse natural language in Clojure and ClojureScript

* <a id="ruby">**Ruby**</a> | [Back to Top](#contents)
  * Kevin Dias's [A collection of Natural Language Processing (NLP) Ruby libraries, tools and software](https://github.com/diasks2/ruby-nlp)
  * [Practical Natural Language Processing done in Ruby](https://github.com/arbox/nlp-with-ruby)
* <a id="rust">**Rust**</a>
  * [whatlang](https://github.com/greyblake/whatlang-rs) — Natural language recognition library based on trigrams
  * [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) - A production ready library for intent parsing

### Services
APIs with higher level functionality such as NER, Topic tagging and so on | [Back to Top](#contents)

* [Wit-ai](https://github.com/wit-ai/wit) - Natural Language Interface for apps and devices
* [IBM Watson's Natural Language Understanding](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs) - API and Github demo 
* [Amazon Comprehend](https://aws.amazon.com/comprehend/) - NLP and ML suite covers most common tasks like NER, tagging, and sentiment analysis
* [Google Cloud Natural Language API](https://cloud.google.com/natural-language/) - Syntax Analysis, NER, Sentiment Analysis, and Content tagging in atleast 9 languages include English and Chinese (Simplified and Traditional).
* [ParallelDots](https://www.paralleldots.com/text-analysis-apis) - State of the art Text Analysis API Service ranging from Sentiment Analysis to Intent Analysis
* [Microsoft Cognitive Service](https://www.microsoft.com/cognitive-services/en-us/text-analytics-api)
* [TextRazor](https://www.textrazor.com/)
* [Rosette](https://www.rosette.com/)

## Techniques

### Text Embeddings

[Back to Top](#contents)

Text embeddings allow deep learning to be effective on smaller datasets. These are often first inputs to a deep learning archiectures and most popular way of transfer learning in NLP. Embeddings are simply vectors or a more generically, real valued representations of strings. Word embeddings are considered a great starting point for most deep NLP tasks. 

The most popular names in word embeddings are word2vec by Google (Mikolov) and GloVe by Stanford (Pennington, Socher and Manning). fastText seems to be a fairly popular for multi-lingual sub-word embeddings. 


#### word2vec and GloVe
**Don't use word2vec, don't use GloVe**. Use fastText vectors, which are much better from the same authors. word2vec was introduced by [T. Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. when he was with Google. 
Performs well on word similarity and analogy tasks. | [Back to Top](#contents)

* [Word2Vec Official Implementation](https://code.google.com/p/word2vec/)
* [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
Chris Olah (2014), Beginner friendly blog explaining word2vec
* [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)
* [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), [Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/), [gensim's Review of word2vec](http://rare-technologies.com/making-sense-of-word2vec/)
* [Word2Vec Resources on Github](https://github.com/clulab/nlp-reading-group/wiki/Word2Vec-Resources)

GloVe was introduced by Pennington, Socher, Manning from Stanford in 2014 as a statistical approximation to word embeddings. The word vectors are created by matrix factorizations of word-word co-occurence matrices here  | [Back to Top](#contents)

* [GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf). Creates word vectors and relates word2vec to matrix factorizations 
* [Glove source code and training data](http://nlp.stanford.edu/projects/glove/)

#### fastText
fastText by Mikolov (from Facebook) supports sub-word embeddings in more than 200 languages. This allows it to work with out of vocabulary words as well. It captures language morphology well. It also supports a supervised classification mechanism | [Back to Top](#contents)

  * [fastText on Github](https://github.com/facebookresearch/fastText) - for efficient learning of word representations and sentence classification
  * [Pre-trained Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html) in several languages
  * [arXiv: Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), [arXiv: Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759), and [arXiv: FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651) were released as part of this project
  * [Unofficial Python Wrapper for fastText on Github](https://github.com/vrasneur/pyfasttext/)

#### Other Text Embeddings

[Back to Top](#contents)

* [Pre-trained word embeddings for WSJ corpus](https://github.com/ai-ku/wvec) by Koc AI-Lab
* [HLBL language model](http://metaoptimize.com/projects/wordreprs/) by Turian
* [Real-valued vector "embeddings"](http://www.cis.upenn.edu/~ungar/eigenwords/) by Dhillon
* [Improving Word Representations Via Global Context And Multiple Word Prototypes](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes) by Huang
* [Dependency based word embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [sense2vec](http://arxiv.org/abs/1511.06388) - on word sense disambiguation
* [Infinite Dimensional Word Embeddings](http://arxiv.org/abs/1511.05392) - new
* [Skip Thought Vectors](http://arxiv.org/abs/1506.06726) - word representation method
* [Adaptive skip-gram](http://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties
* [Sequence to Sequence Learning](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - word vectors for machine translation
* [Improving distributional similarity with lessons learned from word embeddings](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/viewFile/570/124)
* [Deep Contextualized Word Represenations](https://arxiv.org/abs/1802.05365) - [PyTorch](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) - [TF Implementation](https://github.com/allenai/bilm-tf)


### Thought Vectors
Thought vectors are numeric representations for sentences, paragraphs, and documents.  The following papers are listed in order of date published, each one replaces the last as the state of the art in sentiment analysis | [Back to Top](#contents)

* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)
Socher et al. 2013.  Introduces Recursive Neural Tensor Network.  Uses a parse tree.
* [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree. Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)
* [Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks. Uses a parse tree.
* [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)
Tai et al. 2015  Introduces Tree LSTM. Uses a parse tree.
* [Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)
Dai, Le 2015 "With pretraining, we are able to train long short term memory recurrent networks up to a few hundred
timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."

### Machine Translation

[Back to Top](#contents)

* Google Research's [blog post](https://research.googleblog.com/2017/07/building-your-own-neural-machine.html) for neural machine translation using encoder-decoder architecture with seq2seq models. [Tensorflow Code here](https://github.com/tensorflow/nmt)
  * [seq2seq tensorflow tutorial](http://tensorflow.org/tutorials/seq2seq/index.html)
* Prof Graham Neubig's Neural Machine Translation [tutorial in Perl](https://github.com/neubig/nmt-tips)
* [arXiv: Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf) Sutskever, Vinyals, Le 2014 proved the effectiivenss of **LSTM** for Machine Translation. Check their ([nips presentation](http://research.microsoft.com/apps/video/?id=239083))
* [arXiv: Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf)
Bahdanau, Cho 2014 introduced the **attention mechanism** in NLP
* [arXiv: A Convolutional encoder model for neural machine translation](https://arxiv.org/pdf/1611.02344.pdf) by Gehring et al, 2017. The paper is from Facebook AI research and its code is available [here](https://github.com/facebookresearch/fairseq)
* [Convolutional Sequence to Sequence learning](https://arxiv.org/pdf/1705.03122.pdf) by Gehring et al, 2017. The paper is from Facebook AI research and its code is available [here](https://github.com/facebookresearch/fairseq)
* [Convolutional over Recurrent Encoder for neural machine translation](https://ufal.mff.cuni.cz/pbml/108/art-dakwale-monz.pdf) by Dakwale and Monz from University of Amsterdam compare the CNNs with a recurrent neural network with additional convolutonal layers
* Open Source code: [OpenNMT](http://opennmt.net/) is an open source initiative for neural machine translation and neural sequence modeling. [PyTorch](https://github.com/OpenNMT/OpenNMT-py), [Tensorflow](https://github.com/OpenNMT/OpenNMT-tf) and the original [LuaTorch](https://github.com/OpenNMT/OpenNMT) implementation 

### Dialogs and Conversational 

[Back to Top](#contents)
* [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)
Sordoni 2015. Generates responses to tweets. 
  * Uses [Recurrent Neural Network Language Model (RLM) architecture of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf). [Code of RNNLM Toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/index.html)
  * RNNLM Tutorial: [Implementing RNN Language Models by Denny Britz](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
* [Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.
* [arXiv: A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf) Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses

### Memory and Attention Models 

[Back to Top](#contents)
Some are courtesy [andrewt3000/DL4NLP](https://github.com/andrewt3000/DL4NLP)

* Interactive tutorial on [Augmented RNNs](http://www.distill.pub/2016/augmented-rnns/) including Attention and Memory networks
* [Annotated Transformer](http://nlp.seas.harvard.edu//2018/04/03/attention.html) from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) work explains Tranformer implementation in line by line detail. Both links highly recommended.  
* [Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf) Weston et. al 2014
* [End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf) Sukhbaatar et. al 2015
Memory networks are implemented in [MemNN](https://github.com/facebook/MemNN).  Attempts to solve task of reason attention and memory
* [Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)
* [Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf), Graves et al. 2014
* [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf), Joulin, Mikolov 2015 
* [Stack RNN source code](https://github.com/facebook/Stack-RNN) and [blog post](https://research.facebook.com/blog/1642778845966521/inferring-algorithmic-patterns-with-stack/)


### Natural Language Understanding

[Back to Top](#contents)

* [Neural autocoder for paragraphs and documents](http://arxiv.org/abs/1506.01057) - LSTM representation
* [LSTM over tree structures](http://arxiv.org/abs/1503.04881)
* [Low-Dimensional Embeddings of Logic](http://www.aclweb.org/anthology/W/W14/W14-2409.pdf)
* Tutorial on Markov Logic Networks ([based on this paper](http://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf))
* [Distant Supervision for Cancer Pathway Extraction From Text](http://research.microsoft.com/en-us/um/people/hoifung/papers/psb15.pdf)
* [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
* [Retrofitting word vectors to semantic lexicons](http://www.cs.cmu.edu/~mfaruqui/papers/naacl15-retrofitting.pdf)
* [Unsupervised Learning of the Morphology of a Natural Language](http://www.mitpressjournals.org/doi/pdfplus/10.1162/089120101750300490)
* [Computational Grounded Cognition: a new alliance between grounded cognition and computational modelling](http://journal.frontiersin.org/article/10.3389/fpsyg.2012.00612/full)
* [Learning the Structure of Biomedical Relation Extractions](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004216)
* [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf) by T. Mikolov, 2012. * [Slides on the same here](http://www.fit.vutbr.cz/~imikolov/rnnlm/google.pdf)

### Named Entity Recognition

[Back to Top](#contents)

* [A survey of named entity recognition and classification](http://nlp.cs.nyu.edu/sekine/papers/li07.pdf)
* [Benchmarking the extraction and disambiguation of named entities on the semantic web](http://www.lrec-conf.org/proceedings/lrec2014/pdf/176_Paper.pdf)
* [Knowledge base population: Successful approaches and challenges](http://www.aclweb.org/anthology/P11-1115)
* [SpeedRead: A fast named entity recognition Pipeline](http://arxiv.org/pdf/1301.2857.pdf)

### Question Answering and Knowledge Extraction

[Back to Top](#contents)

* [Markov Logic Networks for Natural Language Question Answering](http://arxiv.org/pdf/1507.03045v1.pdf)
* [Template-Based Information Extraction without the Templates](http://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
* [Relation extraction with matrix factorization and universal schemas](http://www.anthology.aclweb.org/N/N13/N13-1008.pdf)
* [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](http://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
* [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340) - DeepMind paper
* [DrQA: Open Domain Question Answering](https://github.com/facebookresearch/DrQA) by facebook on Wikipedia data
* [Relation Extraction with Matrix Factorization and Universal Schemas](http://www.riedelcastro.org//publications/papers/riedel13relation.pdf)
* [Towards a Formal Distributional Semantics: Simulating Logical Calculi with Tensors](http://www.aclweb.org/anthology/S13-1001)
* [Presentation slides for MLN tutorial](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt)
* [Presentation slides for QA applications of MLNs](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf)
* [Presentation slides](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf)

### Text Summarization

[Back to Top](#contents)

* [awesome-text-summarization](https://github.com/mathsyouth/awesome-text-summarization) - curated list of resources in text summarization.
* [Example blogpost](https://medium.com/@Currie32/text-summarization-with-amazon-reviews-41801c2210b) uses Amazon food reviews for text summarization. Code on [Github here](https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews).
* [TextRank- bringing order into text](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) by Mihalcea and Tarau. Code on [Github here](https://github.com/ceteri/pytextrank)
* [Modelling compressions with Discourse constraints](http://jamesclarke.net/media/papers/clarke-lapata-emnlp07.pdf) by Clarke and Zapata provides a discourse informed model for summarization and subtitle generation.
* [Deep Recurrent Generative Decoder model for Abstractive Text Summarization](https://arxiv.org/pdf/1708.00625v1.pdf) by Li et al, 2017 uses a sequence-to-sequence oriented encoder-decoder model equipped with a deep recurrent generative decoder.
* [A Semantic Relevance Based Neural Network for Text Summarization and Text Simplification](https://arxiv.org/pdf/1710.02318v1.pdf) by Ma and Sun, 2017 uses a gated attention enocder-decoder for text summarization.
* [TextSum](https://github.com/tensorflow/models/tree/master/research/textsum) implementation from Tensorflow

### Text Classification

[Back to Top](#contents)
* [Brightmart/text_classification](https://github.com/brightmart/text_classification) has a list of all text classification models with their respective scores, trainings,explanations and their Python implementations.
* [Facebook's fasttext](https://github.com/facebookresearch/fastText) is a library for text embeddings and text classification
* [Convolutional Neural Networks for Sentence Classfication](https://arxiv.org/pdf/1408.5882v2.pdf) by Kim Yoon is now regarded as the standard baseline for text classification architecture. 
* [Using a CNN for text classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) by Denny Britz uses the same dataset as Kim Yoon's paper(mentioned above). The code implementation can be found [here](https://github.com/dennybritz/cnn-text-classification-tf).
* [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626v3.pdf) by Zhang et al uses CNN and compares them with the traditional text classification models. Its Lua implementation can be found [here](https://github.com/zhangxiangxiao/Crepe).

## Datasets

[Back to Top](#contents)

* [nlp-datasets](https://github.com/niderhoff/nlp-datasets) great collection of nlp datasets

## Implementations of various models

[Back to Top](#contents)

* [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch) has Pytorch implementations of various deep NLP models used in CS224n(Stanford) in the form of Jupyter notebooks.The models are aimed for those who are acquainted with Pytorch.

## Multilingual NLP Frameworks

[Back to Top](#contents)
* [UDPipe](https://github.com/ufal/udpipe) : Trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U file. Primarily written in C++, offers a fast and reliable solution for multilingual NLP processing.
* [NLP-Cube](https://github.com/adobe/NLP-Cube) : Natural Language Processing Pipeline - Sentence Splitting, Tokenization, Lemmatization, Part-of-speech Tagging and Dependency Parsing. New platform, written in Python with Dynet 2.0. Offers standalone (CLI/Python bindings) and server functionality (REST API).

## NLP in Korean

[Back to Top](#contents)

### Libraries

  * [KoNLPy](http://konlpy.org) - Python package for Korean natural language processing.
  * [Mecab (Korean)](http://eunjeon.blogspot.com/) - C++ library for Korean NLP
  * [KoalaNLP](https://nearbydelta.github.io/KoalaNLP/) - Scala library for Korean Natural Language Processing.
  * [KoNLP](https://cran.r-project.org/web/packages/KoNLP/index.html) - R package for Korean Natural language processing

### Blogs and Tutorials 

* [dsindex's blog](http://dsindex.github.io/)
* [Kangwon University's NLP course in Korean](http://cs.kangwon.ac.kr/~leeck/NLP/)

### Datasets

* [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus)- A corpus from the Korea Advanced Institute of Science and Technology in Korean.
* [Naver Sentiment Movie Corpus in Korean](https://github.com/e9t/nsmc/)
* [Chosun Ilbo archive](http://srchdb1.chosun.com/pdf/i_archive/) - dataset in Korean from one of the major newspapers in South Korea, the Chosun Ilbo.

## NLP in Arabic

[Back to Top](#contents)

### Libraries

* [goarabic](https://github.com/01walid/goarabic)-  Go package for Arabic text processing
* [jsastem](https://github.com/ejtaal/jsastem) - Javascript for Arabic stemming
* [PyArabic](https://pypi.python.org/pypi/PyArabic/0.4) - Python libraries for Arabic

### Datasets
* [Multidomain Datasets](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces) - Largest Available Multi-Domain Resources for Arabic Sentiment Analysis
* [LABR](https://github.com/mohamedadaly/labr) - LArge Arabic Book Reviews dataset
* [Arabic Stopwords](https://github.com/mohataher/arabic-stop-words) - A list of Arabic stopwords from various resources

## NLP in Chinese

[Back to Top](#contents)

### Libraries

 * [jieba](https://github.com/fxsjy/jieba#jieba-1) - Python package for Words Segmentation Utilities in Chinese
 * [SnowNLP](https://github.com/isnowfy/snownlp) - Python package for Chinese NLP
 * [FudanNLP](https://github.com/FudanNLP/fnlp)- Java library for Chinese text processing

## NLP in German

 * [German-NLP](https://github.com/adbar/German-NLP) - Curated list of open-access/open-source/off-the-shelf resources and tools developed with a particular focus on German
 
## NLP in Spanish

[Back to Top](#contents)

### Corpora

* [Columbian Political Speeches](https://github.com/dav009/LatinamericanTextResources)
* [Copenhagen Treebank](http://code.google.com/p/copenhagen-dependency-treebank/)
* [Reuters Corpora RCV2](http://trec.nist.gov/data/reuters/reuters.html)
* [Spanish Billion words corpus with Word2Vec embeddings](http://crscardellino.me/SBWCE/)

## NLP in Indic languages

[Back to Top](#contents)

### Hindi

### Corpora and Treebanks

* [Hindi Dependency Treebank](http://ltrc.iiit.ac.in/treebank_H2014/) - A multi-representational multi-layered treebank for Hindi and Urdu
* [Universal Dependencies Treebank in Hindi](http://universaldependencies.org/treebanks/hi/index.html)
     * [Parallel Universal Dependencies Treebank in Hindi](http://universaldependencies.org/treebanks/hi_pud/index.html) - A smaller part of the above-mentioned treebank.
     
## NLP in Thai

[Back to Top](#contents)

### Libraries

* [PyThaiNLP](https://github.com/wannaphongcom/pythainlp) - Thai NLP in Python Package
* [JTCC](https://github.com/wittawatj/jtcc)- A character cluster library in Java
* [CutKum](https://github.com/pucktada/cutkum) - Word segmentation with deep learning in TensorFlow
* [Thai Language Toolkit](https://pypi.python.org/pypi/tltk/) - Based on a paper by Wirote Aroonmanakun in 2002 with included dataset
* [SynThai](https://github.com/KenjiroAI/SynThai)- Word segmentation and POS tagging using deep learning in Python

### Corpora

* [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm) - A text corpus with 5 million words with word segmentation
* [Prime Minister 29](https://github.com/PyThaiNLP/lexicon-thai/tree/master/thai-corpus/Prime%20Minister%2029)- Dataset containing speeches of the current Prime Minister of Thailand

## NLP in Vietnamese

[Back to Top](#contents)

### Libraries

* [underthesea](https://github.com/magizbox/underthesea) - Vietnamese NLP Toolkit
* [vn.vitk](https://github.com/phuonglh/vn.vitk) - A Vietnamese Text Processing Toolkit
* [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) - A Vietnamese natural language processing toolkit

### Corpora

* [Vietnamese treebank](https://vlsp.hpda.vn/demo/?page=resources&lang=en) - 10,000 sentences for the constituency parsing task
* [BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf) - a Vietnamese Dependency Treebank
* [UD_Vietnamese](https://github.com/UniversalDependencies/UD_Vietnamese-VTB) - Vietnamese Universal Dependency Treebank
* [VIVOS](https://ailab.hcmus.edu.vn/vivos/) - a free Vietnamese speech corpus consisting of 15 hours of recording speech by AILab
* [VNTQcorpus(big).txt](http://viet.jnlp.org/download-du-lieu-tu-vung-corpus) - 1.75 million sentences in news

### Other Languages 
   * Russian: [pymorphy2](https://github.com/kmike/pymorphy2) - a good pos-tagger for Russian
   * Asian Languages: Thai, Lao, Chinese, Japanese, and Korean [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html) implementation in ElasticSearch
   * Ancient Languages: [CLTK](https://github.com/cltk/cltk): The Classical Language Toolkit is a Python library and collection of texts for doing NLP in ancient languages
   * Dutch: [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
   * Hebrew: [NLPH_Resources](https://github.com/NLPH/NLPH_Resources) - A collection of papers, corpora and linguistic resources for NLP in Hebrew

## Credits
Awesome NLP was seeded with curated content from the lot of repositories, some of which are listed below | [Back to Top](#contents)
* [ai-reading-list](https://github.com/m0nologuer/AI-reading-list)
* [nlp-reading-group](https://github.com/clulab/nlp-reading-group/wiki/Fall-2015-Reading-Schedule/_edit)
* [awesome-spanish-nlp](https://github.com/dav009/awesome-spanish-nlp)
* [jjangsangy's awesome-nlp](https://gist.github.com/jjangsangy/8759f163bc3558779c46)
* [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning/edit/master/README.md)
* [DL4NLP](https://github.com/andrewt3000/DL4NLP)

[Back to Top](#contents)
