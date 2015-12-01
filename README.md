# awesome-nlp
A curated list of resources dedicated to Natural Language Processing


Maintainers - [Keon Kim](http://github.com/keonkim)

## Contributing
Please feel free to [pull requests](https://github.com/keonkim/awesome-nlp/pulls), email Keon Kim (keon.kim@nyu.edu) to add links.


## Table of Contents

 - [Tutorials and Courses](#tutorials-and-courses)
   - [videos](#videos)
 - [Codes](#codes)
   - [Implemendations](#implementations)
   - [Libraries](#libraries)
     - [Node.js](#node.js)
     - [Java](#java)
     - [Python](#python)
     - [C++](#c++)
 - [Articles](#articles)
   - [Word Vectors](#word-vectors)
   - [General Natural Language Processing](#general-natural-langauge-processing)
   - [Supplementary Materials](#supplementary-materials)
 - [Blogs](#blogs)
 - [Multilingual](#multillingual)
   - [Spanish](#spanish)
 - [Credits](#credits)


## Tutorials and Courses

* Tensor Flow Tutorial on [Seq2Seq](http://www.tensorflow.org/tutorials/seq2seq/index.html) Models

### videos
* [Stanford's Coursera Course](https://www.coursera.org/course/nlp) on NLP from basics
* [Intro to Natural Language Processing](https://www.coursera.org/course/nlpintro) on Coursera by U of Michigan
* [Intro to Artificial Intelligence](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) course on Udacity which also covers NLP

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
* **Node.js** - Node.js Libaries for NLP
  * [Natural](https://github.com/NaturalNode/natural) - general natural language facilities for node
* **Python** - Python NLP Libraries
  * [Natural Language Toolkit (NLTK)](http://www.nltk.org/)
* **C++** - C++ Libraries
  * [Mecab (Japanese)](http://taku910.github.io/mecab/)
  * [Mecab (Korean)](http://eunjeon.blogspot.com/)
* **Java** - Java NLP Libraries
  * [Stanford NLP](http://nlp.stanford.edu/software/index.shtml)
  * [Word2vec in Java](http://deeplearning4j.org/word2vec.html)
  
## Articles

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
