# 令人讚嘆的自然語言處理 [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> 專門用於自然語言處理的精選資源列表

![Awesome NLP Logo](/images/logo.jpg)

> * 原文地址：[令人讚嘆的自然語言處理](https://github.com/keon/awesome-nlp)
> * 原文作者：[Keon](https://github.com/keon), [Martin](https://github.com/outpark), [Nirant](https://github.com/NirantK), [Dhruv](https://github.com/the-ethan-hunt)
> * 翻譯：[NeroCube](https://github.com/NeroCube)

_請在提交之前閱讀 [貢獻指南](contributing.md) 。請隨時創建 [拉取請求](https://github.com/keonkim/awesome-nlp/pulls)._

## 內容

* [研究摘要和趨勢](#研究摘要和趨勢)
* [教學](#教學)
  * [閱讀內容](#閱讀內容)
  * [影片和課程](#影片和課程)
  * [書籍](#書籍)
* [函式庫](#函式庫)
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
* [服務](#服務)
* [註釋工具](#註釋工具)
* [資料集](#資料集)
* [自然語言處理-韓文](#自然語言處理-韓文)
* [自然語言處理-阿拉伯語](#自然語言處理-阿拉伯語)
* [自然語言處理-中文](#自然語言處理-中文)
* [自然語言處理-德文](#自然語言處理-德文)
* [自然語言處理-西班牙語](#自然語言處理-西班牙語)
* [自然語言處理-印度語](#自然語言處理-印度語)
* [自然語言處理-泰語](#自然語言處理-泰語)
* [自然語言處理-丹麥語](#自然語言處理-丹麥語)
* [自然語言處理-越南語](#自然語言處理-越南語)
* [自然語言處理-印度尼西亞](#自然語言處理-印度尼西亞)
* [其他語言](#其他語言)
* [貢獻](#貢獻)

## 研究摘要和趨勢

* [自然語言處理-概述](https://nlpoverview.com/) 是應用於自然語言深度學習技術的最新概述，包括理論，實現，應用和最先進的結果。對於研究人員來說，這是一個偉大的Deep NLP簡介。 
* [自然語言處理-進展](https://nlpprogress.com/) 追隨自然語言處理的進展，包括資料集和常見自然語言處理任務的當前最新技術。
* [自然語言處理的 ImageNet 時刻已經到來](https://thegradient.pub/nlp-imagenet/)
* [ACL 2018 亮點: 在更具挑戰性的設置中理解表示和評估](http://ruder.io/acl-2018-highlights/)
* [ACL 2017 的四個深度學習趨勢。第一部分：語言結構和詞語嵌入](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-1.html)
* [ACL 2017 的四個深度學習趨勢。第二部分：可解釋性和注意力](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-2.html)
* [2017 年 EMNLP 的亮點：激動人心的資料集，集群的回歸與其他更多！](http://blog.aylien.com/highlights-emnlp-2017-exciting-datasets-return-clusters/)
* [深度學習自然語言處理 (NLP): 進展與趨勢](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)
* [自然語言生成的現狀調查](https://arxiv.org/abs/1703.09902)

## 教學
[返回頂部](#內容)

### 閱讀內容

通用機器學習

* 來自 Google 高級創意工程師 Jason 的[機器學習 101](https://docs.google.com/presentation/d/1kSuQyW5DTnkVaZEjGYCkfOxvzCqGEFzWBy4e9Uedd9k/edit?usp=sharing) ，為工程師和管理階層解釋機器學習。
* a16z [AI 劇本](https://aiplaybook.a16z.com/) 是一個很好的鏈接，可以轉發給您的經理或演示內容。
* [繼器學習部落格](https://bmcfee.github.io/#home) by Brian McFee
* [Ruder's 部落格](http://ruder.io/#open) 由 [Sebastian Ruder](https://twitter.com/seb_ruder) 進行評論得最好的自然語言處理研究。

自然語言處理介紹與指南

* [理解和實施自然語言處理](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/) 的終極指南。
* [Hackernoon 的自然語言處理簡介](https://hackernoon.com/learning-ai-if-you-suck-at-math-p7-the-magic-of-natural-language-processing-f3819a689386) 適用於那些賞味數學的人-用他們自己的話來說。
* [Vik Paruchari 的自然語言處理教學](http://www.vikparuchuri.com/blog/natural-language-processing-tutorial/)
* [自然語言處理: 一份簡介](https://academic.oup.com/jamia/article/18/5/544/829676) 來自牛津大學。
* [使用 Pytorch 進行自然語言處理的深度學習](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
* [動手做 NLTK 教學](https://github.com/hb20007/hands-on-nltk-tutorial) - 以  
 Jupyter 筆記本形式的實踐 NLTK 教學。

部落格與簡報

* 部落格: [深度學習, 自然語言處理, 與呈現法](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* 部落格: [圖解 BERT, ELMo, 與 co. (自然語言處理是如何破解遷移學習的)](https://jalammar.github.io/illustrated-bert/) 與 [圖解轉換器](https://jalammar.github.io/illustrated-transformer/)
* 部落格: Hal Daumé III 的[自然語言處理](https://nlpers.blogspot.com/)
* [Radim Řehůřek 的教學](https://radimrehurek.com/gensim/tutorial.html) 使用 Python 與 [gensim](https://radimrehurek.com/gensim/index.html) 處理語言語料庫。
* [arXiv: 自然語言處理 (大部分) 來自 Scratch](https://arxiv.org/pdf/1103.0398.pdf)
* [Karpathy 的遞歸神經網絡的不合理有效性](https://karpathy.github.io/2015/05/21/rnn-effectiveness)

### 影片和課程

#### 深度學習與自然語言處理

用於自然語言處理的詞嵌入, 遞歸神經網絡, 長短期記憶神經網絡與卷積神經網路 | [返回頂部](#內容)

* Udacity 的[人工智慧入門](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) 課程涉及到自然語言處理。
* Udacity 的[深度學習](https://udacity.com/course/deep-learning--ud730) 使用Tensorflow 使用深度學習的 NLP 任務的部分（包括 Word2Vec，RNN的 和 LSTMs）。
* 牛津大學的[深度自然語言處理](https://github.com/oxford-cs-deepnlp-2017/lectures)有影片，演講投影片和閱讀素材。
* 斯坦福大學的[自然語言處理深度學習 (cs224-n)](https://web.stanford.edu/class/cs224n/) 由 Richard Socher 和 Christopher Manning 完成。
* Coursera 的[自然語言處理](https://www.coursera.org/learn/language-processing) 由國立研究大學高等經濟學院完成。
* 卡內基梅隆大學的語言技術研究所[自然語言處理的神經網路](http://phontron.com/class/nn4nlp2017/)。

#### 經典自然語言處理

自然語言處理的貝葉斯，統計和語言學方法| | [返回頂部](#內容)

* [統計機器翻譯](http://mt-class.org) - 機器翻譯課程，具有很棒的作業和投影片。
* [使用 Python 3 進行 NLTK 自然語言處理](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) 由 Harrison Kinsley(sentdex) 使用 NLTK 程式碼實現的好教學。
* 由 Jordan Boyd-Graber 在馬里蘭大學的[計算語言學 I](https://www.youtube.com/playlist?list=PLegWUnz91WfuPebLI97-WueAP90JO-15i)講座。
* 由 Yandex 數據學院的[深度自然語言處理課程](https://github.com/yandexdataschool/nlp_course)涵蓋從文本嵌入到機器翻譯的重要思想，包括序列建模，語言模型等。

### 書籍

* Dan Jurafsy 教授的[語音和語言處理](https://web.stanford.edu/~jurafsky/slp3/)
* [R 中的文字探勘](https://www.tidytextmining.com)
* [Python 的自然語言處理](https://www.nltk.org/book/)

## 函式庫

[返回頂部](#內容)

* <a id="node-js">**Node.js and Javascript** - 用於自然語言的 Node.js 函式庫</a> | [返回頂部](#內容)
  * [Twitter-text](https://github.com/twitter/twitter-text) - 使用 JavaScript 實現的 Twitter 文本處理庫。
  * [Knwl.js](https://github.com/benhmoore/Knwl.js) - JS中的自然語言處理器。
  * [Retext](https://github.com/retextjs/retext) - 用於分析和操縱自然語言的可​​擴展系統。
  * [NLP Compromise](https://github.com/spencermountain/compromise) - 瀏覽器中的自然語言處理。
  * [Natural](https://github.com/NaturalNode/natural) - 節點的一般自然語言設施。
  - [Poplar](https://github.com/synyi/poplar) - 一種基於 Web 的自然語言處理註釋工具（NLP）。

* <a id="python"> **Python** - 用於自然語言的 Python 函式庫</a> | [返回頂部](#內容)

  * [TextBlob](http://textblob.readthedocs.org/) - 為專研常見的自然語言處理（NLP）任務提供一致的 API。 站在[自然語言工具包 (NLTK)](https://www.nltk.org/) 和 [模式](https://github.com/clips/pattern)膀上，並與兩者很好地配合 :+1:
  * [spaCy](https://github.com/explosion/spaCy) - 使用 Python 與 Cython 產業強度的自然語言處理  :+1:
    * [textacy](https://github.com/chartbeat-labs/textacy) -  在spaCy上構建的更高級別的自然與儼處理。
  * [gensim](https://radimrehurek.com/gensim/index.html) - 用於從純文本進行無監督語義建模的 函式庫 :+1:
  * [scattertext](https://github.com/JasonKessler/scattertext) - 用於生成語料庫之間語言差異的 d3 可視化的 Python 函式庫。
  * [AllenNLP](https://github.com/allenai/allennlp) - 一個架構在 PyTorch 上的自然語言處理函式庫，用於開發各種語言任務最先進的深度學習模型。
  * [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - 自然語言處理研究工具包設計來支援快速建立更好的數據加載器，詞向量加載器，神經網路層表示，常見的自然語言處理指標（如BLEU）原型。
  * [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - 文本處理工具和包裝 (例如： Vowpal Wabbit)
  * [PyNLPl](https://github.com/proycon/pynlpl) - Python 自然語言處理函式庫. 適用於 Python 的通用自然語言處理函式庫。 還包含一些用於解析常見自然語言處理格式的特定模塊, 最常見的是用於 [FoLiA](https://proycon.github.io/folia/)，還包括 ARPA 語言模型，Moses 短語表，GIZA ++對齊。

  * [jPTDP](https://github.com/datquocnguyen/jPTDP) - 用於聯合詞性（POS）標記和依賴性解析的工具包。jPTDP 提供40多種語言的預訓練模型。
  * [BigARTM](https://github.com/bigartm/bigartm) - 一個用於主題建模的快速函式庫。
  * [Snips NLU](https://github.com/snipsco/snips-nlu) - 用於意圖解析的產品就緒函式庫。
  * [Chazutsu](https://github.com/chakki-works/chazutsu) - 用於下載和解析標準自然語言處理研究數據集的函式庫。
  * [Word Forms](https://github.com/gutfeeling/word_forms) - Word forms 可以準確生成所有可能的英語單詞形式。
  * [Multilingual Latent Dirichlet Allocation (LDA)](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA) - 一種多語言和可擴展的文檔聚類管道。
  * [NLP Architect](https://github.com/NervanaSystems/nlp-architect) - 用於探索 NLP 和 NLU 最先進的深度學習拓撲和技術的函式庫。
  * [Flair](https://github.com/zalandoresearch/flair) - 一個非常簡單的框架，用於在 PyTorch 上構建最先進的多語言 NLP。包括 BERT，ELMo 和 Flair 嵌入。
  * [Kashgari](https://github.com/BrikerMan/Kashgari) - 簡單的，基於 Keras 的多語言自然語言處理框架，允許您在5分鐘內構建模型，用於命名實體識別（NER），詞性標註（PoS）和文本分類任務。 包括 BERT 和 word2vec 嵌入。


* <a id="c++">**C++** - C++ 函式庫</a> | [返回頂部](#內容)
  * [MIT 資訊提取工具包 ](https://github.com/mit-nlp/MITIE) - 用於命名實體識別和關係提取的 C，C++ 和Python 工具。
  * [CRF++](https://taku910.github.io/crfpp/) - 條件隨機場（CRF）的開源專案，用於實現分割/標記順序數據和其他自然語言處理任務。
  * [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite 實現用於標記順序數據的條件隨機字段（CRF）。
  * [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP 自然語言解析器（也稱為 Charniak-Johnson 解析器）
  * [colibri-core](https://github.com/proycon/colibri-core) - C++ 函式庫，命令行工具和 Python 綁定用於快速且內存有效的方式提取和使用基本語言結構，如 n-gram 和 skipgrams。
  * [ucto](https://github.com/LanguageMachines/ucto) - 適用於各種語言的基於 Unicode 的常規表達式標記生成器。工具和 C++函式庫。支持 FoLiA 格式。
  * [libfolia](https://github.com/LanguageMachines/libfolia) - 用於 [FoLiA 格式](https://proycon.github.io/folia/)的 C++ 函式庫。
  * [frog](https://github.com/LanguageMachines/frog) - 為荷蘭語開發的基於內存的自然語言處理套件：PoS 標記器，lemmatiser，依賴解析器，NER，淺層解析器，形態分析器。
  * [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) 是一個 C++ 數據科學工具包，可以幫助挖掘大文本數據。
  * [Mecab (日文)](https://taku910.github.io/mecab/)
  * [Moses](http://statmt.org/moses/)
  * [StarSpace](https://github.com/facebookresearch/StarSpace) - 一個來自 Facebook 的函式庫用於創建單詞級，段級，文檔級和文本分類的嵌入

* <a id="java">**Java** - Java 自然語言處理函式庫</a> | [返回頂部](#內容)
  * [斯坦福大學 NLP](https://nlp.stanford.edu/software/index.shtml)
  * [OpenNLP](https://opennlp.apache.org/)
  * [NLP4J](https://emorynlp.github.io/nlp4j/)
  * [Java 中的 Word2vec](https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec)
  * [ReVerb](https://github.com/knowitall/reverb/) Web-Scale 開放信息提取。
  * [OpenRegex](https://github.com/knowitall/openregex) 一種高效靈活的基於 token 的正則表達式語言和引擎。
  * [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - 在伊利諾伊大學的認知計算組開發的核心函式庫。
  * [MALLET](http://mallet.cs.umass.edu/) - 用於 LanguagE Toolkit 的機器學習 - 用於統計自然語言處理，文檔分類，聚類，主題建模，資訊提取和其他機器學習應用程序的文本包。
  * [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - 一個穩健的 POS 標記工具包（包括 Java 和 Python）以及40多種語言的預訓練模型。
  
* <a id="kotlin">**Kotlin** - Kotlin 自然語言處理函式庫</a> | [返回頂部](#內容)
  * [Lingua](https://github.com/pemistahl/lingua/) 適用於 Kotlin 和 Java 的語言檢測函式庫，適用於長文本和短文本。
  * [Kotidgy](https://github.com/meiblorn/kotidgy) — 一種用 Kotlin 編寫基於索引的文本數據生成器。
  
* <a id="scala">**Scala** - Scala 自然語言處理函式庫</a> | [返回頂部](#內容)
  * [Saul](https://github.com/CogComp/saul) - 用於開發自然語言處理系統的函式庫，包括內置模塊，如 SRL，POS 等。
  * [ATR4S](https://github.com/ispras/atr4s) - 具有最先進的[自動術語識別](https://en.wikipedia.org/wiki/Terminology_extraction)方法的工具包。
  * [tm](https://github.com/ispras/tm) - 基於正則化多語言 [PLSA](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis) 的主題建模實現。 
  * [word2vec-scala](https://github.com/Refefer/word2vec-scala) - word2vec 模型的 Scala 接口; 包括對詞距離和詞類比等向量的操作。
  * [Epic](https://github.com/dlwh/epic) - Epic 是一個用 Scala 編寫的高性能統計解析器，以及用於構建複雜結構化預測模型的框架。

* <a id="R">**R** - R 自然語言處理函式庫</a> | [返回頂部](#內容)
  * [text2vec](https://github.com/dselivanov/text2vec) -  R 中的快速矢量化，主題建模，距離和 GloVe 字嵌入。
  * [wordVectors](https://github.com/bmschmidt/wordVectors) - 用於創建和探索 word2vec 和其他單詞嵌入模型的 R 包。
  * [RMallet](https://github.com/mimno/RMallet) - 與 Java 機器學習工具 MALLET 接口的 R 包。
  * [dfr-browser](https://github.com/agoldst/dfr-browser) -  創建用於在 Web 瀏覽器中瀏覽文本主題模型的 d3 可視化。
  * [dfrtopics](https://github.com/agoldst/dfrtopics) - 用於探索文本主題模型的 R 包。
  * [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - 使用Word Sense Disambiguation 和 WordNet Reader 的情感分類。
  * [jProcessing](https://github.com/kevincobain2000/jProcessing) - 日本自然語言處理庫，具有日語情感分類。

* <a id="clojure">**Clojure**</a> | [返回頂部](#內容)
  * [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Clojure 中的自然語言處理（opennlp）。
  * [Infections-clj](https://github.com/r0man/inflections-clj) - 用於 Clojure 和 ClojureScript 的類似 Rails 的變形函式庫。
  * [postagga](https://github.com/fekr/postagga) - 用於解析 Clojure 和 ClojureScript 中的自然語言的函式庫。

* <a id="ruby">**Ruby**</a> | [返回頂部](#內容)
  * Kevin Dias 的 [自然語言處理（NLP）Ruby 函式庫，工具和軟件的集合](https://github.com/diasks2/ruby-nlp)
  * [Ruby 中實用的自然語言處理](https://github.com/arbox/nlp-with-ruby)

* <a id="rust">**Rust**</a> | [返回頂部](#內容)
  * [whatlang](https://github.com/greyblake/whatlang-rs) — 基於三元組的自然語言識別函式庫。
  - [snips-nlu-rs](https://github.com/snipsco/snips-nlu-rs) - 用於意圖解析的生產就緒等級函示庫。

### 服務

自然語言處理作為具有更高級功能的 API，例如 NER，主題標記等 | [返回頂部](#內容)

- [Wit-ai](https://github.com/wit-ai/wit) - 應用程序和設備的自然語言界面。
- [IBM Watson 的自然語意理解](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs) - API 和 Github 演示。
- [Amazon 理解](https://aws.amazon.com/comprehend/) - NLP 和 ML 套件涵蓋了最常見的任務，如 NER，標記和情感分析。
- [Google 雲端自然語言 API](https://cloud.google.com/natural-language/) - 至少9種語言的語法分析，NER，情感分析和內容標記包括英語和中文（簡體和繁體）。
- [ParallelDots](https://www.paralleldots.com/text-analysis-apis) - 高層次文本分析 API 服務，從情感分析到意圖分析。
- [Microsoft 認知服務](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
- [TextRazor](https://www.textrazor.com/)
- [Rosette](https://www.rosette.com/)
- [Textalytic](https://www.textalytic.com) - 瀏覽器中的自然語言處理，包括情感分析，命名實體提取，POS標記，詞頻，主題建模，文字雲等。

### 註釋工具

- [GATE](https://gate.ac.uk/overview.html) - 通用架構和文本工程已有15年歷史，免費開源。
- [Anafora](https://github.com/weitechen/anafora) 是免費的開源，基於 Web 的原始文本註釋工具。
- [brat](https://brat.nlplab.org/) - brat 快速註解工具是一個用於協作文本註釋的在線環境。
- [tagtog](https://www.tagtog.net/), 需花 $。
- [prodigy](https://prodi.gy/) 是一個由主動學習驅動的註釋工具，需花 $。
- [LightTag](https://lighttag.io) - 為團隊提供託管和管理的文本註釋工具，需花 $。

## 技術

### 文本嵌入

[返回頂部](#內容)

文本嵌入允許深度學習在較小的數據集上有效。這些通常是深入學習的第一步輸入和自然語言處理中最流行的遷移學習方式。嵌入只是簡單的向量，比實際值的字符串表示更為通用的方式。Word嵌入被認為是大多數深度NLP任務的一個很好的起點。

單詞嵌入中最流行的名字是 Google（Mikolov）的 word2vec 和史丹佛的 PenVe（Pennington，Socher 和Manning）。fastText 似乎是一種非常流行的多語言子詞嵌入。

#### 詞嵌入

[返回頂部](#內容)

|嵌入 |論文| 組織| gensim - 培訓支援 |部落格|
|---|---|---|---|---|
|word2vec|[官方實作](https://code.google.com/archive/p/word2vec/), T.Mikolove et al. 2013. 分散式詞語表達及其組合性。[pdf](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |Google|是 :heavy_check_mark:|  colah 在[深度學習，自然語言處理和陳述](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)中的視覺會解釋; gensim 的[理解 word2vec](https://rare-technologies.com/making-sense-of-word2vec) |
|GloVe|Jeffrey Pennington, Richard Socher 與 Christopher D. Manning. 2014. GloVe: 全局向量的字詞表示 [pdf](https://nlp.stanford.edu/pubs/glove.pdf)|史丹佛|否 :negative_squared_cross_mark:|acoyler 的 [GloVe 早報](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/) |
|fastText|[官方實作](https://github.com/facebookresearch/fastText), T. Mikolov et al. 2017. 使用子詞資訊豐富單詞向量。 [pdf](https://arxiv.org/abs/1607.04606)|Facebook|是 :heavy_check_mark:|[Fasttext: 深入解析](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)|

給初學者的筆記:

- 經驗法則: **fastText >> GloVe > word2vec**
- 你可以找到許多語言[預訓練 fasttext 向量](https://fasttext.cc/docs/en/pretrained-vectors.html)。
- 如果你對 word2vec 和 GloVe 背後的邏輯和直覺感興趣: [詞向量的驚人力量](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)並很好地介紹這些主題。
- [arXiv: 高效文本分類的錦囊妙方](https://arxiv.org/abs/1607.01759), 與 [arXiv: FastText.zip: 壓縮文本分類模型](https://arxiv.org/abs/1612.03651) 作為 fasttext 的一部分發布。


#### 基於句子和語言模型的詞嵌入

[返回頂部](#內容)

- _ElMo_ 從[深度情境詞表示](https://arxiv.org/abs/1802.05365) - [PyTorch 實作](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) - [TF 實作](https://github.com/allenai/bilm-tf)
- _ULimFit_ Jeremy Howard 與 Sebastian Ruder 的[通用語言模型進行文本分類微調](https://arxiv.org/abs/1801.06146)
- _InferSent_ facebook 的 [自然語言推論資料的通用語句表示監督是學習](https://arxiv.org/abs/1705.02364)
- _CoVe_ from [在翻譯中學習: 情境詞相量](https://arxiv.org/abs/1708.00107)
- _來自[文件與句子的分散式表達](https://cs.stanford.edu/~quocle/paragraph_vector.pdf). 參閱 [gensim 的 doc2vec 教學](https://rare-technologies.com/doc2vec-tutorial/)
- [sense2vec](https://arxiv.org/abs/1511.06388) - 關於詞義消歧。
- [跳過思考象量](https://arxiv.org/abs/1506.06726) - 單詞表示方法。
- [自適應 skip-gram](https://arxiv.org/abs/1502.07257) - 類似的方法，具有自適應屬性。
- [序列到序列學習](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - 機器翻譯的詞向量。

### 回答問題與知識提取

[返回頂部](#內容)

- Facebook 透過維基百科 [DrQA: 打開領域為題解答](https://github.com/facebookresearch/DrQA) 
- DocQA: AllenAI 的[簡單而有效的多段閱讀理解](https://github.com/allenai/document-qa)
- [用於自然語言問答的馬爾可夫邏輯網絡](https://arxiv.org/pdf/1507.03045v1.pdf)
- [基於模板的資訊提取沒有用到模板](https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
- [矩陣分解與通用模式的關係提取](https://www.anthology.aclweb.org/N/N13/N13-1008.pdf)
- [Privee：自動分析Web隱私策略的體系結構](https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
- [教學機器閱讀和理解](https://arxiv.org/abs/1506.03340) - DeepMind paper
- [走向形式分佈語義：用張量模擬邏輯演算](https://www.aclweb.org/anthology/S13-1001)
- [MLN 教學的演示投影片](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt)
- [MLNs 的 QA 應用演示投影片](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf)
- [演示投影片](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf)

## 資料集

[返回頂部](#內容)

- [nlp-datasets](https://github.com/niderhoff/nlp-datasets) 很好的自然語言資料集集合

## 多語言自然語言處理框架

[返回頂部](#內容)

- [UDPipe](https://github.com/ufal/udpipe) 是一個可訓練的管道，用於標記，標記，解釋和解析通用樹庫和其他 CoNLL-U 文件。主要用 C++ 編寫，為多語言NLP處理提供快速可靠的解決方案。
- [NLP-Cube](https://github.com/adobe/NLP-Cube) : 自然語言處理流水線 - 句子分裂，標記化，詞形還原，詞性標註和依賴性分析。用 Dynet 2.0 用 Python 編寫的新平台。提供獨立（CLI / Python 綁定）和服務器功能（REST API）。

## 自然語言處理-韓文

[返回頂部](#內容)

### 函式庫

- [KoNLPy](http://konlpy.org) - 用於韓語自然語言處理的Python包。
- [Mecab (Korean)](https://eunjeon.blogspot.com/) - 韓文的自然語言處理 C++ 函式庫
- [KoalaNLP](https://koalanlp.github.io/koalanlp/) - 韓國自然語言處理的 Scala 函式庫。
- [KoNLP](https://cran.r-project.org/package=KoNLP) - 韓文的自然語言處理 R 包。

### 部落格與教學

- [dsindex 的部落格](https://dsindex.github.io/)
- [韓國江原大學的自然語言處理課程](http://cs.kangwon.ac.kr/~leeck/NLP/)

### 資料集

- [KAIST 語料庫](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus) - 韓國高等科學技術研究所的語料庫。
- [韓國 Naver 情感電影語料庫](https://github.com/e9t/nsmc/)
- [朝鮮日報檔案館](http://srchdb1.chosun.com/pdf/i_archive/) - 來自韓國主要報紙之一的朝鮮日報的韓文數據集。

## 自然語言處理-阿拉伯語

[返回頂部](#內容)

### 函式庫

- [goarabic](https://github.com/01walid/goarabic) - Go包用於阿拉伯語文本處理。
- [jsastem](https://github.com/ejtaal/jsastem) - 用於阿拉伯詞幹的Javascript。
- [PyArabic](https://pypi.org/project/PyArabic/) - 阿拉伯語的 Python 函式庫。

### 資料集

- [多域數據集](https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces) - 阿拉伯語情感分析的最大可用多域資源。
- [LABR](https://github.com/mohamedadaly/labr) - LArge阿拉伯書籍評論數據集。
- [Arabic 停用詞](https://github.com/mohataher/arabic-stop-words) - 來自各種資源的阿拉伯語停用詞列表。

## 自然語言處理-中文

[返回頂部](#內容)

### 函式庫

- [jieba](https://github.com/fxsjy/jieba#jieba-1) - 中文詞彙分割實用程序的 Python 包。
- [SnowNLP](https://github.com/isnowfy/snownlp) - 中文自然語言處理 Python 包。
- [FudanNLP](https://github.com/FudanNLP/fnlp) - 用於中文文本處理的 Java 函式庫。

## 自然語言處理-德文

[返回頂部](#內容)

- [德文-自然語言處理](https://github.com/adbar/German-NLP) - 開發的開放式訪問/開源/現成資源和工具列表，特別關注德語。
 
## 自然語言處理-西班牙語

[返回頂部](#內容)

### 資料

- [哥倫比亞政治演說](https://github.com/dav009/LatinamericanTextResources)
- [哥本哈根樹庫](https://mbkromann.github.io/copenhagen-dependency-treebank/)
- [西班牙語十億字語料庫與 Word2Vec 嵌入](https://github.com/crscardellino/sbwce)

## 自然語言處理-印度語

[返回頂部](#內容)

### 印地語

### 資料, 文集與樹庫

- [印地語依賴樹庫](https://ltrc.iiit.ac.in/treebank_H2014/) - 印地語和烏爾都語的多代表性多層樹庫。
- [在印地語的普遍依賴性樹庫](https://universaldependencies.org/treebanks/hi_hdtb/index.html)
  - [並行通用依賴樹庫印地語](http://universaldependencies.org/treebanks/hi_pud/index.html) - 上述樹庫的一小部分。

## 自然語言處理-泰語

[返回頂部](#內容)

### 函式庫

- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) - Python 包中的泰語自然語言處理。
- [JTCC](https://github.com/wittawatj/jtcc) - Java 中的字符集群庫。
- [CutKum](https://github.com/pucktada/cutkum) - 在 TensorFlow 中使用深度學習進行分詞。
- [泰語工具包](https://pypi.python.org/pypi/tltk/) - 基於 Wirote Aroonmanakun 於2002年撰寫的一篇論文，其中包括數據集。
- [SynThai](https://github.com/KenjiroAI/SynThai) - 在 Python 中使用深度學習進行分詞和 POS 標記。

### 資料

- [Inter-BEST](https://www.nectec.or.th/corpus/index.php?league=pm) - 具有500萬個單詞分詞的文本語料庫。
- [Prime Minister 29](https://github.com/PyThaiNLP/lexicon-thai/tree/master/thai-corpus/Prime%20Minister%2029) - 數據集包含現任泰國總理的演講。


## 自然語言處理-丹麥語 

[返回頂部](#內容)

- [丹麥的命名實體識別](https://github.com/ITUnlp/daner)

## 自然語言處理-越南語

[返回頂部](#內容)

### 函式庫

- [underthesea](https://github.com/undertheseanlp/underthesea) - 越南自然語言處理工具包。
- [vn.vitk](https://github.com/phuonglh/vn.vitk) - 越南文本處理工具包。
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) - 越南自然語言處理工具包。

### 資料

- [越南樹庫](https://vlsp.hpda.vn/demo/?page=resources&lang=en) - 選區解析任務的10,000個句子。
- [BKTreeBank](https://arxiv.org/pdf/1710.05519.pdf) -  越南依賴樹庫。
- [UD_Vietnamese](https://github.com/UniversalDependencies/UD_Vietnamese-VTB) - 越南通用依賴樹庫。
- [VIVOS](https://ailab.hcmus.edu.vn/vivos/) - 一個免費的越南語言語料庫，由 AILab 的15小時錄音講話組成。
- [VNTQcorpus(big).txt](http://viet.jnlp.org/download-du-lieu-tu-vung-corpus) - 新聞中的175萬句話。

## 自然語言處理-印度尼西亞

[返回頂部](#內容)

### 資料集
- [ILPS](http://ilps.science.uva.nl/resources/bahasa/) 的Kompas 和 Tempo 系列。
- [用於PoS標記的PANL10N](http://www.panl10n.net/english/outputs/Indonesia/UI/0802/UI-1M-tagged.zip): 39K句子和900K字標記。
- [用於PoS標記的IDN](https://github.com/famrashel/idn-tagged-corpus): 該語料庫包含10K個句子和250K個單詞標記。
- [印度尼西亞樹庫](https://github.com/famrashel/idn-treebank)和 [普遍依賴 - 印度尼西亞語](https://github.com/UniversalDependencies/UD_Indonesian-GSD)
- [IndoSum](https://github.com/kata-ai/indosum) 用於文本摘要和分類。
- [Wordnet-Bahasa](http://wn-msa.sourceforge.net/) - 大型，免費的語義詞典。

### 函式庫與嵌入
- 自然語言工具包 [bahasa](https://github.com/kangfend/bahasa)
- [印尼語嵌入](https://github.com/galuhsahid/indonesian-word-embedding)
- 預訓練的訓練 [印尼 fastText 文本嵌入](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.id.zip) 的維基百科。

## 其他語言 

[返回頂部](#內容)

- 俄語: [pymorphy2](https://github.com/kmike/pymorphy2) - - 俄語好的詞性標記。
- 亞洲語言: ElasticSearch 中的泰語，老撾語，中文，日語和韓語 [ICU Tokenizer](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu-tokenizer.html) 實現。
- 古代語言: [CLTK](https://github.com/cltk/cltk): 古典語言工具包是一個 Python 函式庫和用於在古代語言中進行自然語言處理的文本集合。
- Dutch: [python-frog](https://github.com/proycon/python-frog) - Python 綁定到 Frog，一個荷蘭語的自然語言處理套件。（pos 標記，詞形還原，依賴解析，NER
- 希伯來語: [NLPH_Resources](https://github.com/NLPH/NLPH_Resources) - 希伯來語自然語言處理的論文，語料庫和語言資源的集合。

## 貢獻

初始策展人和來源的[貢獻](./CREDITS.md)。
