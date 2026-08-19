# 📘 The NLP Handbook — From Zero to Transformers

*A single-file, code-first reference for Natural Language Processing: theory in a few bullet points, working code right below it. Built to be searched (Ctrl/Cmd+F), not read cover to cover.*

> **Last verified against library documentation:** August 2026.
> NLP libraries move fast. Version-sensitive gotchas are called out in **⚠️ boxes** throughout — most notably, **Hugging Face Transformers hit v5.0 in January 2026**, which removed TensorFlow/Flax support and several convenience pipelines. See §5.9 for the full migration note. If a snippet ever errors out, it's almost always a version mismatch — check the library's changelog first.

---

## How to use this handbook

- It's organized as **10 parts**, roughly following the order you'd actually learn NLP: preprocessing → classical representations → tasks → transformers → modern LLM-era tooling → evaluation.
- Every technique includes **at least one working code snippet** using the library's built-in functions — copy, paste, adjust variable names, done.
- 🔧 = practical tip / gotcha. ⚠️ = version-specific breaking change to know about. 📎 = pointer to official docs.
- Code assumes Python 3.10+ in a virtual environment.

---

## Table of Contents

**[Part 0 — Setup & Installation](#part-0--setup--installation)**

**[Part 1 — Core Libraries](#part-1--core-libraries)**
[1.1 NLTK](#11-nltk) · [1.2 spaCy](#12-spacy) · [1.3 TextBlob](#13-textblob) · [1.4 Hugging Face Transformers](#14-hugging-face-transformers) · [1.5 Gensim](#15-gensim) · [1.6 Other Notable NLP Libraries](#16-other-notable-nlp-libraries)

**[Part 2 — Text Preprocessing](#part-2--text-preprocessing)**
[2.1 Tokenization](#21-tokenization) · [2.2 Stopword Removal](#22-stopword-removal) · [2.3 Punctuation & Noise Removal](#23-punctuation--noise-removal) · [2.4 Stemming](#24-stemming) · [2.5 Lemmatization](#25-lemmatization) · [2.6 Text Normalization](#26-text-normalization) · [2.7 Part-of-Speech (POS) Tagging](#27-part-of-speech-pos-tagging) · [2.8 Parsing](#28-parsing-dependency--constituency) · [2.9 Regex Cheatsheet for Text Cleaning](#29-regex-cheatsheet-for-text-cleaning) · [2.10 Full Preprocessing Pipeline](#210-full-reusable-preprocessing-pipeline)

**[Part 3 — Text Representation & Embeddings](#part-3--text-representation--embeddings)**
[3.1 One-Hot Encoding](#31-one-hot-encoding) · [3.2 Bag of Words](#32-bag-of-words-bow) · [3.3 TF-IDF](#33-term-frequency-inverse-document-frequency-tf-idf) · [3.4 N-Gram Language Modeling](#34-n-gram-language-modeling) · [3.5 LSA](#35-latent-semantic-analysis-lsa) · [3.6 LDA](#36-latent-dirichlet-allocation-lda) · [3.7 Word2Vec](#37-word2vec) · [3.8 GloVe](#38-glove) · [3.9 fastText](#39-fasttext) · [3.10 ELMo](#310-elmo-contextual-embeddings-legacy) · [3.11 BERT Embeddings](#311-bert-embeddings) · [3.12 Doc2Vec](#312-doc2vec) · [3.13 Sentence-BERT](#313-sentence-embeddings-sentence-bert) · [3.14 RoBERTa / DistilBERT](#314-roberta--distilbert) · [3.15 Comparison Table](#315-embedding-methods-comparison-table)

**[Part 4 — Core NLP Tasks](#part-4--core-nlp-tasks)**
[4.1 Text Classification](#41-text-classification) · [4.2 Named Entity Recognition](#42-named-entity-recognition-ner) · [4.3 Text Summarization](#43-text-summarization) · [4.4 Sentiment Analysis](#44-sentiment-analysis) · [4.5 Machine Translation](#45-machine-translation) · [4.6 Question Answering](#46-question-answering) · [4.7 Topic Modeling Recap + BERTopic](#47-topic-modeling-recap--bertopic) · [4.8 Text Generation](#48-text-generation) · [4.9 Semantic Similarity & Search](#49-semantic-similarity--search) · [4.10 Coreference Resolution](#410-coreference-resolution) · [4.11 Text Clustering](#411-text-clustering)

**[Part 5 — Transformer Architectures Deep-Dive](#part-5--transformer-architectures-deep-dive)**
[5.1 Attention & Self-Attention](#51-attention--self-attention-the-math) · [5.2 The Transformer Architecture](#52-the-full-transformer-architecture) · [5.3 Subword Tokenization (BPE/WordPiece/SentencePiece)](#53-subword-tokenization-algorithms) · [5.4 The BERT Family](#54-the-bert-family-encoder-only) · [5.5 The GPT Family](#55-the-gpt-family-decoder-only) · [5.6 T5 / BART (Encoder-Decoder)](#56-t5--bart-encoder-decoder) · [5.7 pipeline() Task Reference](#57-the-pipeline-quick-task-reference) · [5.8 Manual AutoTokenizer/AutoModel Workflow](#58-manual-autotokenizerautomodel-workflow) · [5.9 Fine-Tuning with Trainer](#59-fine-tuning-with-the-trainer-api) · [5.10 Parameter-Efficient Fine-Tuning (LoRA/PEFT)](#510-parameter-efficient-fine-tuning-loraqlorapeft)

**[Part 6 — Modern NLP: The LLM Era](#part-6--modern-nlp-the-llm-era)**
[6.1 Prompt Engineering Basics](#61-prompt-engineering-basics) · [6.2 Retrieval-Augmented Generation (RAG)](#62-retrieval-augmented-generation-rag) · [6.3 Vector Databases](#63-vector-databases--similarity-search-at-scale) · [6.4 Calling LLM APIs](#64-calling-llm-apis-for-nlp-tasks) · [6.5 LangChain Quick Reference](#65-langchain-quick-reference)

**[Part 7 — Evaluation Metrics](#part-7--evaluation-metrics)**
[7.1 Classification Metrics](#71-classification-metrics) · [7.2 BLEU / ROUGE / METEOR](#72-bleu--rouge--meteor-generationtranslationsummarization) · [7.3 Perplexity](#73-perplexity-language-model-quality) · [7.4 BERTScore](#74-bertscore-embedding-based-evaluation)

**[Part 8 — End-to-End Worked Example](#part-8--end-to-end-worked-example)**

**[Part 9 — Quick-Reference Cheat Sheets](#part-9--quick-reference-cheat-sheets)**

**[Part 10 — Resources & Further Reading](#part-10--resources--further-reading)**

---

## Part 0 — Setup & Installation

```bash
# Create an isolated environment first (strongly recommended)
python -m venv nlp-env
source nlp-env/bin/activate        # Windows: nlp-env\Scripts\activate

# --- Core stack ---
pip install nltk spacy textblob gensim

# --- Transformers / deep learning stack (PyTorch backend — see §5.9 for why) ---
pip install torch transformers datasets evaluate accelerate

# --- Embeddings, topic modeling, retrieval ---
pip install sentence-transformers bertopic faiss-cpu

# --- Fine-tuning ---
pip install peft trl bitsandbytes

# --- Classical ML / metrics ---
pip install scikit-learn pandas numpy matplotlib seaborn

# --- Misc but frequently useful ---
pip install beautifulsoup4 contractions pyspellchecker rapidfuzz langdetect wordcloud emoji sacrebleu rouge_score

# spaCy models (download separately from the pip package)
python -m spacy download en_core_web_sm     # small, fast, CPU-friendly (~13MB)
python -m spacy download en_core_web_trf    # RoBERTa-based, most accurate, needs a GPU for speed
```

```python
# NLTK's data (corpora, tokenizer models, taggers) is downloaded separately at runtime.
# 🔧 NLTK 3.9+ split several resources by language / renamed them — download ALL of these
# up front so you don't hit a LookupError mid-pipeline (old tutorials often list only 'punkt').
import nltk

for pkg in [
    "punkt", "punkt_tab",                      # sentence/word tokenizer models
    "stopwords",                                # stopword lists
    "wordnet", "omw-1.4",                       # WordNet + Open Multilingual WordNet (lemmatizer)
    "averaged_perceptron_tagger",               # POS tagger (older resource name)
    "averaged_perceptron_tagger_eng",           # POS tagger (current, language-suffixed name)
    "maxent_ne_chunker", "maxent_ne_chunker_tab",  # rule-based NER chunker
    "words",                                    # word list used by the NE chunker
    "vader_lexicon",                            # VADER sentiment lexicon
]:
    nltk.download(pkg, quiet=True)
```

```python
# TextBlob also needs its own corpora on first use
# (run once in a terminal): python -m textblob.download_corpora
```

**🔧 Debugging tip:** if NLTK raises `LookupError: Resource xyz not found`, the error message
literally tells you the exact `nltk.download('xyz')` call to run — just copy it. Resource names
have changed across NLTK versions (e.g. `punkt` → `punkt_tab`, `averaged_perceptron_tagger` →
`averaged_perceptron_tagger_eng`), so don't trust an old blog post's download list blindly.

**GPU check** (matters a lot once you reach Part 5):
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```


---

## Part 1 — Core Libraries

### 1.1 NLTK

The original Swiss-army knife of Python NLP (since 2001). Best for **teaching, prototyping, and classical/rule-based techniques**. Slower than spaCy for production pipelines, but unmatched breadth of corpora and algorithms in one place.

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "NLTK is a leading platform for building Python programs to work with human language data."

print(word_tokenize(text))                       # word tokens
print(sent_tokenize(text))                        # sentence tokens
print(stopwords.words("english")[:10])            # first 10 English stopwords
print(PorterStemmer().stem("running"))             # 'run'
print(WordNetLemmatizer().lemmatize("running", pos="v"))  # 'run'
```

📎 Docs: https://www.nltk.org/

---

### 1.2 spaCy

**Industrial-strength NLP.** Object-oriented, fast (Cython under the hood), ships pretrained pipelines per language, and is the go-to choice when you need production-grade tokenization, POS tagging, dependency parsing, and NER out of the box.

```python
import spacy

nlp = spacy.load("en_core_web_sm")   # or en_core_web_trf for the transformer-based pipeline
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.is_stop)

for ent in doc.ents:
    print(ent.text, ent.label_)

# Visualize dependency parse / entities (renders in Jupyter; use style="dep" or "ent")
from spacy import displacy
displacy.render(doc, style="dep", jupyter=True)
# Outside Jupyter: displacy.serve(doc, style="dep") then open http://localhost:5000
```

**Pipeline components** available on `nlp.pipe_names` typically include: `tok2vec`, `tagger`, `parser`, `attribute_ruler`, `lemmatizer`, `ner`. You can disable components you don't need for a speed boost:
```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
# Or process many documents efficiently:
for doc in nlp.pipe(list_of_texts, batch_size=50, n_process=4):
    ...
```

📎 Docs: https://spacy.io/usage

---

### 1.3 TextBlob

A friendly, high-level wrapper around NLTK + pattern.en. Great for **quick sentiment analysis, spelling correction, and simple noun-phrase extraction** — not built for heavy production workloads.

```python
from textblob import TextBlob

blob = TextBlob("Textblob is amazingly simple to use. What great fun!")

print(blob.sentiment)          # Sentiment(polarity=0.55, subjectivity=0.7) — polarity: -1..1, subjectivity: 0..1
print(blob.tags)                # POS tags: [('Textblob', 'NN'), ('is', 'VBZ'), ...]
print(blob.noun_phrases)        # WordList(['textblob'])
print(blob.words)               # tokenized words
print(blob.sentences)           # tokenized sentences
print(TextBlob("I havv goood speling").correct())   # 'I have good spelling' — naive spell correction
```

🔧 TextBlob's `.translate()` / `.detect_language()` methods relied on an **unofficial** Google
Translate endpoint and have been unreliable for years — for translation or language detection,
use `transformers` (§4.5), the `langdetect` / `langid` packages, or a proper translation API instead.

📎 Docs: https://textblob.readthedocs.io/

---

### 1.4 Hugging Face Transformers

**The standard library for pretrained transformer models** — BERT, GPT-family, T5, RoBERTa, Llama, and thousands of others, all behind one unified API. This is where "classical NLP" hands off to "modern NLP."

```python
from transformers import pipeline

# The pipeline() function is the fastest way to get state-of-the-art results in one line.
classifier = pipeline("sentiment-analysis")
print(classifier("I absolutely love how simple this API is!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

⚠️ **Transformers hit v5.0 in January 2026** — a major breaking release. If you copy code from an
older tutorial (or this changes again), see the full migration note in **§5.9**. The short version:
PyTorch is now the *only* backend (TensorFlow/Flax support was removed), and a few single-purpose
`pipeline()` shortcuts (`summarization`, `translation`, `question-answering`) were retired in favor
of instruction-tuned `text-generation` models. Both the classic and modern approaches are shown
throughout Part 4 so this handbook works whichever version you're pinned to.

📎 Docs: https://huggingface.co/docs/transformers

---

### 1.5 Gensim

Built specifically for **unsupervised topic modeling, document similarity, and word embeddings** on large corpora, with a memory-efficient, streaming-friendly design (corpora don't need to fit in RAM).

```python
from gensim import corpora
from gensim.models import Word2Vec, LdaModel

sentences = [["nlp", "is", "fun"], ["gensim", "trains", "word", "embeddings"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv.most_similar("nlp", topn=3))
```

⚠️ **Gensim 4.0 (2021) was a breaking release** whose effects are still visible in most tutorials
you'll find online — and it removed `gensim.summarization` entirely (unmaintained). Full details
and migration notes are in §3.7 and §4.3. Current stable release: **Gensim 4.4.x**.

📎 Docs: https://radimrehurek.com/gensim/

---

### 1.6 Other Notable NLP Libraries

| Library | What it's for | Notes |
|---|---|---|
| **Stanza** (Stanford NLP) | Neural pipelines (tokenize → NER) for 70+ languages | `pip install stanza`; `import stanza; stanza.download('en'); nlp = stanza.Pipeline('en')` |
| **Flair** | Easy contextual-embedding NER/classification/POS | `pip install flair`; great "just works" NER out of the box |
| **sentence-transformers** | Sentence/document embeddings, semantic search, reranking | See §3.13 — the standard for embedding-based retrieval |
| **BERTopic** | Modern transformer-based topic modeling | See §4.7 — largely superseded classic LDA workflows |
| **KeyBERT** | Keyword/keyphrase extraction using BERT embeddings | `pip install keybert`; `KeyBERT().extract_keywords(doc)` |
| **textacy** | Higher-level NLP utilities built on top of spaCy | Corpus stats, readability scores, n-gram extraction |
| **TextAttack** | Adversarial attacks & data augmentation for NLP models | Used to stress-test text classifiers |
| **presidio** (Microsoft) | PII detection & anonymization | `pip install presidio-analyzer presidio-anonymizer` |
| **fastText** (Facebook, standalone) | Extremely fast text classification + subword vectors | `pip install fasttext`; C++ core, great for production |
| **langdetect** / **langid** / **fastText lid.176** | Language identification | `pip install langdetect` — `from langdetect import detect` |
| **rapidfuzz** (successor to fuzzywuzzy) | Fast fuzzy string matching / edit distance | `pip install rapidfuzz`; `fuzz.ratio(str1, str2)` |
| **wordcloud** | Word-cloud visualizations | Popular for quick EDA on text corpora |
| **PyTorch / TensorFlow text utilities** | Low-level tensor ops for custom architectures | `torchtext`, `tf.strings`, `tf.keras.layers.TextVectorization` |
| **AllenNLP** | Research-oriented NLP framework from AI2 | ⚠️ Archived/no longer actively maintained — avoid for new projects |
| **NeuralCoref** | Coreference resolution add-on for spaCy | ⚠️ Unmaintained, built for spaCy 2.x only — use `coreferee` or `fastcoref` instead (§4.10) |
| **Stanford CoreNLP** | Java-based classical NLP toolkit | Access from Python via `stanza`'s CoreNLP client or a REST wrapper |
| **LangChain / LlamaIndex** | LLM orchestration: RAG, agents, chains | See Part 6 — glue code for building LLM-powered applications |


---

## Part 2 — Text Preprocessing

### 2.1 Tokenization

Splitting raw text into smaller units — words, subwords, or sentences — before anything else can happen.

```python
# --- Word tokenization ---
from nltk.tokenize import word_tokenize, wordpunct_tokenize, TreebankWordTokenizer
text = "Don't hesitate to email support@example.com, it's free!"

print(word_tokenize(text))            # handles contractions & punctuation sensibly
print(wordpunct_tokenize(text))       # splits ALL punctuation into separate tokens
print(TreebankWordTokenizer().tokenize(text))  # Penn Treebank conventions (word_tokenize uses this internally)

# --- Sentence tokenization ---
from nltk.tokenize import sent_tokenize
print(sent_tokenize("Dr. Smith went to Washington. He arrived on Jan. 5th."))
# ['Dr. Smith went to Washington.', 'He arrived on Jan. 5th.']  -- correctly avoids splitting on 'Dr.'/'Jan.'

# --- spaCy tokenization (rule + exception based, very robust) ---
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print([token.text for token in doc])
print([sent.text for sent in doc.sents])   # sentence tokenization via the parser

# --- Regex-based custom tokenization ---
from nltk.tokenize import regexp_tokenize
print(regexp_tokenize(text, pattern=r"\w+"))   # words only, no punctuation

# --- Whitespace-only (fastest, dumbest) ---
print(text.split())
```

**Subword tokenization** (what transformer models actually use — BPE, WordPiece, SentencePiece) is covered in depth in §5.3, since it's tightly coupled to how transformer vocabularies are built.

```python
# Quick preview: a transformer tokenizer splits unknown/rare words into known sub-pieces
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tok.tokenize("Transformerization is unbelievably powerful"))
# ['transformer', '##ization', 'is', 'unbelievable', '##ly', 'powerful']
```

---

### 2.2 Stopword Removal

Stopwords (*the, is, at, which, on…*) carry little semantic weight for many tasks (BoW/TF-IDF classifiers, keyword extraction) but **should usually be kept** for tasks where word order and function words matter (translation, generation, most transformer-based tasks — those models were trained on natural text including stopwords).

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

tokens = ["this", "is", "a", "sample", "sentence", "showing", "stopword", "removal"]
filtered = [w for w in tokens if w.lower() not in stop_words]
print(filtered)   # ['sample', 'sentence', 'showing', 'stopword', 'removal']

# --- spaCy: stopwords are a token attribute ---
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sample sentence showing stopword removal")
print([token.text for token in doc if not token.is_stop])

# --- scikit-learn's built-in English stopword list (used internally by CountVectorizer/TfidfVectorizer) ---
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print(len(ENGLISH_STOP_WORDS))

# --- Custom stopword list (domain-specific — e.g. add 'covid', 'patient' for a medical corpus) ---
custom_stops = stop_words | {"covid", "patient", "said"}
```

🔧 Always inspect the stopword list for your domain — legal/medical/financial text often needs
custom additions (or removals — e.g. "not" is a default stopword but is critical for sentiment).

---

### 2.3 Punctuation & Noise Removal

```python
import string, re

text = "Wow!!! This product is... AMAZING? 10/10 would buy again :) #bestpurchase @brand"

# --- Fast built-in approach: str.translate ---
no_punct = text.translate(str.maketrans("", "", string.punctuation))
print(no_punct)

# --- Regex approach (more control — e.g. keep sentence-ending punctuation) ---
no_punct_regex = re.sub(r"[^\w\s]", "", text)
print(no_punct_regex)

# --- spaCy: punctuation as a token attribute ---
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print([t.text for t in doc if not t.is_punct])

# --- Removing extra whitespace, HTML tags, URLs, mentions, hashtags, digits ---
from bs4 import BeautifulSoup

def strip_html(t):        return BeautifulSoup(t, "html.parser").get_text()
def strip_urls(t):        return re.sub(r"https?://\S+|www\.\S+", "", t)
def strip_mentions(t):    return re.sub(r"@\w+", "", t)
def strip_hashtags(t):    return re.sub(r"#\w+", "", t)
def strip_digits(t):      return re.sub(r"\d+", "", t)
def normalize_spaces(t):  return re.sub(r"\s+", " ", t).strip()
```

---

### 2.4 Stemming

Crude, rule-based suffix chopping to reduce words to a root form — fast, but the output isn't
always a real word (`"studies"` → `"studi"`). Good for search/IR indexing where speed matters more
than linguistic correctness.

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

words = ["running", "runner", "studies", "studying", "national", "generously"]

porter = PorterStemmer()               # the original, most widely used, moderate aggressiveness
snowball = SnowballStemmer("english")  # "Porter2" — improved, supports multiple languages
lancaster = LancasterStemmer()         # very aggressive, can over-stem

for w in words:
    print(f"{w:12} Porter={porter.stem(w):10} Snowball={snowball.stem(w):10} Lancaster={lancaster.stem(w)}")
```

| Stemmer | Aggressiveness | Notes |
|---|---|---|
| Porter | Moderate | Oldest, most conservative, English only |
| Snowball ("Porter2") | Moderate | Improved rules, supports ~15 languages via `SnowballStemmer(<lang>)` |
| Lancaster | Aggressive | Shortest outputs, higher risk of merging unrelated words |

---

### 2.5 Lemmatization

Reduces words to their dictionary base form (**lemma**) using vocabulary + morphological analysis
— slower than stemming but linguistically correct (`"studies"` → `"study"`, `"better"` → `"good"`).
**Requires a POS tag to be accurate** (the lemma of "meeting" differs as a noun vs. verb).

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("studies"))            # 'study'      (default pos='n', noun)
print(lemmatizer.lemmatize("studies", pos="v"))    # 'study'
print(lemmatizer.lemmatize("running", pos="v"))    # 'run'
print(lemmatizer.lemmatize("better", pos="a"))     # 'good'       (pos='a' = adjective)

# Map NLTK's Penn Treebank POS tags to WordNet's simplified tag set for accurate lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"): return wordnet.ADJ
    if treebank_tag.startswith("V"): return wordnet.VERB
    if treebank_tag.startswith("N"): return wordnet.NOUN
    if treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN  # default

import nltk
tagged = nltk.pos_tag(nltk.word_tokenize("The striped bats are hanging on their feet for best"))
lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
print(lemmas)

# --- spaCy: lemmatization is automatic, context-aware, and doesn't need manual POS mapping ---
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The striped bats are hanging on their feet for best")
print([token.lemma_ for token in doc])
```

**Stemming vs. Lemmatization — when to use which:**

| | Stemming | Lemmatization |
|---|---|---|
| Speed | Fast | Slower (needs vocabulary lookup / model) |
| Output | May not be a real word | Always a valid dictionary word |
| Needs POS context? | No | Yes, for full accuracy |
| Best for | Search/IR indexing, quick BoW pipelines | Anything where output readability or grammatical accuracy matters |


---

### 2.6 Text Normalization

The umbrella term for making text consistent before downstream processing: case folding, unicode
normalization, contraction expansion, spelling correction, and noisy-symbol cleanup.

```python
import re, unicodedata

# --- Lowercasing ---
text = "The Quick Brown FOX Jumps!"
print(text.lower())

# --- Unicode normalization (accented chars, curly quotes, ligatures → canonical/ASCII form) ---
text_accented = "café naïve résumé"
normalized = unicodedata.normalize("NFKD", text_accented).encode("ascii", "ignore").decode("utf-8")
print(normalized)   # 'cafe naive resume'

# --- Expanding contractions ---
import contractions
print(contractions.fix("I can't believe it's not butter, y'all wouldn't've known"))
# "I cannot believe it is not butter, you all would not have known"

# --- Spelling correction ---
from spellchecker import SpellChecker    # pip install pyspellchecker
spell = SpellChecker()
misspelled = spell.unknown(["speling", "korrect"])
for word in misspelled:
    print(word, "->", spell.correction(word))

# TextBlob alternative (simpler, less accurate on rare words):
from textblob import TextBlob
print(TextBlob("I havv goood speling").correct())

# --- Removing emojis ---
import emoji
print(emoji.replace_emoji("Great job! 🎉🔥", replace=""))

# --- Number handling ---
import re
print(re.sub(r"\d+", "<NUM>", "I have 3 apples and 42 oranges"))   # placeholder tokens
# or spell out numbers:
from num2words import num2words
print(num2words(42))   # 'forty-two'

# --- Putting it together: a general normalization function ---
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = contractions.fix(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text

print(normalize_text("I CAN'T see the café's website: https://example.com  😀"))
```

---

### 2.7 Part-of-Speech (POS) Tagging

Labeling each token with its grammatical role (noun, verb, adjective, …) — a prerequisite for
accurate lemmatization, parsing, and many information-extraction tasks.

```python
import nltk
tokens = nltk.word_tokenize("The quick brown fox jumps over the lazy dog")
print(nltk.pos_tag(tokens))
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ...]

# --- spaCy: two levels of granularity ---
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")
for token in doc:
    print(token.text, token.pos_, token.tag_)
    # token.pos_ = coarse universal tag (e.g. 'NOUN', 'VERB')
    # token.tag_ = fine-grained Penn Treebank-style tag (e.g. 'NN', 'VBZ')
```

**Common Penn Treebank POS tags** (used by NLTK and spaCy's `.tag_`):

| Tag | Meaning | Tag | Meaning | Tag | Meaning |
|---|---|---|---|---|---|
| NN | Noun, singular | VB | Verb, base form | JJ | Adjective |
| NNS | Noun, plural | VBD | Verb, past tense | JJR | Adjective, comparative |
| NNP | Proper noun, singular | VBG | Verb, gerund/present participle | JJS | Adjective, superlative |
| NNPS | Proper noun, plural | VBN | Verb, past participle | RB | Adverb |
| PRP | Personal pronoun | VBP | Verb, non-3rd person singular present | IN | Preposition/subordinating conj. |
| DT | Determiner | VBZ | Verb, 3rd person singular present | CC | Coordinating conjunction |

---

### 2.8 Parsing (Dependency & Constituency)

**Dependency parsing** identifies grammatical relationships between words (subject, object,
modifier…) — the modern, dominant approach, and what spaCy does natively.

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat")

for token in doc:
    print(f"{token.text:8} --{token.dep_:8}--> {token.head.text}")

displacy.render(doc, style="dep", jupyter=True)   # visual dependency tree (Jupyter)
```

**Constituency parsing** breaks a sentence into nested phrase structures (NP, VP, PP…) — the
older, tree-bank style approach.

```python
# Simple rule-based chunking with NLTK (a lightweight constituency-style parse)
import nltk
sentence = nltk.pos_tag(nltk.word_tokenize("The quick brown fox jumps over the lazy dog"))
grammar = "NP: {<DT>?<JJ>*<NN>}"     # noun phrase = optional determiner + adjectives + noun
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(sentence)
print(tree)
# tree.draw()  # opens a graphical parse tree viewer

# For full neural constituency parsing, use benepar (Berkeley Neural Parser) on top of spaCy:
# pip install benepar; then add "benepar" as a spaCy pipeline component.
```

---

### 2.9 Regex Cheatsheet for Text Cleaning

| Pattern | Matches | Example use |
|---|---|---|
| `\s+` | One or more whitespace chars | Collapse whitespace: `re.sub(r"\s+", " ", text)` |
| `\d+` | One or more digits | Strip numbers |
| `[^\w\s]` | Any non-word, non-space char | Strip punctuation |
| `https?://\S+` | HTTP/HTTPS URLs | Strip links |
| `\S+@\S+\.\S+` | Email-like strings | Strip/extract emails |
| `#\w+` | Hashtags | Strip/extract hashtags |
| `@\w+` | @-mentions | Strip/extract mentions |
| `<[^>]+>` | HTML tags | Strip markup (prefer BeautifulSoup for real HTML) |
| `(.)\1{2,}` | 3+ repeated characters | Squash "sooooo good" → "soo good" |
| `[^\x00-\x7F]+` | Non-ASCII characters | Strip emoji/accents (blunt instrument) |
| `\b[A-Z]{2,}\b` | ALL-CAPS words | Detect shouting/acronyms |
| `^\s*$` | Blank/whitespace-only line | Filter empty lines |

```python
import re
text = "Sooooo good!! Contact me@ test@email.com or visit https://site.com #amazing @friend"

text = re.sub(r"(.)\1{2,}", r"\1\1", text)          # 'Sooooo' -> 'Soo'
text = re.sub(r"https?://\S+", "", text)             # strip URL
text = re.sub(r"\S+@\S+\.\S+", "", text)              # strip email
text = re.sub(r"[#@]\w+", "", text)                   # strip hashtags/mentions
text = re.sub(r"\s+", " ", text).strip()
print(text)
```

---

### 2.10 Full, Reusable Preprocessing Pipeline

A composable function you can paste into any project and enable/disable steps as needed:

```python
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = False,
    remove_punct: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> list[str]:
    if remove_html:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text()
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
    if lowercase:
        text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    tokens = word_tokenize(text)

    if remove_punct:
        tokens = [t for t in tokens if t not in string.punctuation]
    if remove_numbers:
        tokens = [t for t in tokens if not t.isdigit()]
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

sample = "The QUICK brown foxes are running & jumping over 12 lazy dogs! Visit https://example.com"
print(preprocess(sample))
# ['quick', 'brown', 'fox', 'running', 'jumping', 'lazy', 'dog']
```

🔧 **When NOT to preprocess this aggressively:** transformer models (BERT, GPT, etc.) were
pretrained on natural, punctuated, cased text — stripping stopwords/punctuation/case **hurts**
their performance. Heavy preprocessing like this is for classical BoW/TF-IDF/Word2Vec pipelines
(Part 3), not for feeding text into a transformer tokenizer (§5.3).


---

## Part 3 — Text Representation & Embeddings

*How do you turn words into numbers? This part goes roughly in chronological/complexity order:
sparse counting methods → dense static embeddings → contextual transformer embeddings.*

### 3.1 One-Hot Encoding

Each word becomes a binary vector with a single `1` at its vocabulary index. Simple, but produces
huge, sparse vectors with **no notion of similarity** between words (every pair is equidistant).

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

words = np.array(["cat", "dog", "fish", "cat", "bird"]).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
one_hot = encoder.fit_transform(words)
print(encoder.categories_)
print(one_hot)

# Pandas equivalent (handy for categorical/tabular text features)
import pandas as pd
print(pd.get_dummies(["cat", "dog", "fish", "cat", "bird"]))
```

---

### 3.2 Bag of Words (BoW)

Represents a document as an unordered "bag" of word counts. Ignores grammar and word order but is
fast, interpretable, and still a solid baseline for classification.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are great pets",
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())     # vocabulary
print(bow_matrix.toarray())                    # document-term matrix (rows=docs, cols=vocab counts)

# Common tuning knobs:
vectorizer = CountVectorizer(
    max_features=5000,    # keep only the top-N most frequent terms
    min_df=2,              # ignore terms appearing in fewer than 2 documents
    max_df=0.9,             # ignore terms appearing in more than 90% of documents (too common)
    ngram_range=(1, 2),     # unigrams AND bigrams
    stop_words="english",
)
```

---

### 3.3 Term Frequency-Inverse Document Frequency (TF-IDF)

Weighs word counts by how *distinctive* a word is across the corpus — common words (like "the")
get down-weighted, rare-but-relevant words get boosted. The workhorse of classical text search
and a strong classification baseline even today.

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\left(\frac{N}{\text{df}(t)}\right)$$

where `tf(t,d)` = frequency of term *t* in document *d*, `N` = total number of documents, and
`df(t)` = number of documents containing *t*.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are great pets",
]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(corpus)

print(tfidf.get_feature_names_out())
print(tfidf_matrix.toarray().round(3))

# Cosine similarity between documents using their TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(tfidf_matrix))

# Two-step alternative if you already have raw counts (CountVectorizer -> TfidfTransformer)
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
counts = CountVectorizer().fit_transform(corpus)
tfidf_from_counts = TfidfTransformer().fit_transform(counts)
```

---

### 3.4 N-Gram Language Modeling

An n-gram is a contiguous sequence of *n* tokens. N-gram models estimate `P(word | previous n-1 words)`
— the statistical ancestor of modern language models.

```python
from sklearn.feature_extraction.text import CountVectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
print(bigram_vectorizer.fit(["the quick brown fox"]).get_feature_names_out())
# ['brown fox', 'quick brown', 'the quick']

# --- Raw n-grams with NLTK ---
from nltk import ngrams
tokens = "the quick brown fox jumps".split()
print(list(ngrams(tokens, 2)))   # bigrams
print(list(ngrams(tokens, 3)))   # trigrams

# --- Building an actual n-gram LANGUAGE MODEL (with smoothing) using nltk.lm ---
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

text = [["the", "quick", "brown", "fox"], ["the", "quick", "dog"]]
n = 2
train_data, vocab = padded_everygram_pipeline(n, text)

model = Laplace(n)          # Laplace (add-one) smoothing handles unseen n-grams gracefully
model.fit(train_data, vocab)

print(model.score("fox", ["quick"]))    # P(fox | quick)
print(model.generate(5, text_seed=["the"]))   # generate 5 tokens continuing from "the"
```

---

### 3.5 Latent Semantic Analysis (LSA)

Applies **Truncated SVD (dimensionality reduction)** to a TF-IDF matrix to uncover latent
"concepts" — words that tend to co-occur are pulled into the same dimensions. This is how you get
semantic similarity out of a purely sparse, count-based representation.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

corpus = ["The cat sat on the mat", "The dog sat on the log", "Cats and dogs are great pets",
          "Astronomy studies stars and planets", "The telescope observed a distant galaxy"]

tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus)
lsa = TruncatedSVD(n_components=2, random_state=42)
lsa_topics = lsa.fit_transform(tfidf_matrix)
print(lsa_topics.round(3))    # each document's coordinates in the 2 latent "concept" dimensions

# --- Gensim's equivalent: LsiModel ---
from gensim import corpora
from gensim.models import LsiModel

tokenized = [doc.lower().split() for doc in corpus]
dictionary = corpora.Dictionary(tokenized)
bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

lsi = LsiModel(bow_corpus, id2word=dictionary, num_topics=2)
for idx, topic in lsi.print_topics():
    print(idx, topic)
```

---

### 3.6 Latent Dirichlet Allocation (LDA)

A **generative probabilistic model** for topic modeling: assumes each document is a mixture of
topics, and each topic is a distribution over words. Unlike LSA, LDA's topics are genuinely
probabilistic and tend to be more human-interpretable.

```python
from gensim import corpora
from gensim.models import LdaModel

documents = [
    "The cat and dog played in the garden",
    "Stock markets rallied after the earnings report",
    "The kitten chased a ball of yarn",
    "Investors are optimistic about interest rate cuts",
]
tokenized = [doc.lower().split() for doc in documents]

dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]

lda = LdaModel(
    corpus=corpus, id2word=dictionary, num_topics=2,
    passes=10, random_state=42, alpha="auto",
)

for idx, topic in lda.print_topics(num_words=5):
    print(f"Topic {idx}: {topic}")

# Infer the topic distribution for a new/unseen document
new_doc_bow = dictionary.doc2bow("the dog barked at the cat".lower().split())
print(lda.get_document_topics(new_doc_bow))

# --- scikit-learn's equivalent (works directly on a CountVectorizer/BoW matrix) ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

count_matrix = CountVectorizer(stop_words="english").fit_transform(documents)
sk_lda = LatentDirichletAllocation(n_components=2, random_state=42)
sk_lda.fit(count_matrix)

# --- Visualizing topics interactively (very popular for LDA specifically) ---
# pip install pyLDAvis
# import pyLDAvis.gensim_models as gensimvis
# vis = gensimvis.prepare(lda, corpus, dictionary)
# pyLDAvis.display(vis)
```

🔧 For modern topic modeling that leverages transformer embeddings instead of pure word counts,
see **BERTopic** in §4.7 — it generally produces more coherent, less "bag of loosely related
words" topics than classic LDA, especially on shorter documents like tweets or reviews.


### 3.7 Word2Vec

Learns **dense, static** word vectors (typically 100-300 dimensions) by predicting a word from its
context (CBOW) or context from a word (Skip-gram), such that semantically similar words end up
close together in vector space. Famous for vector arithmetic: `king - man + woman ≈ queen`.

⚠️ **Gensim 4.0 breaking change** (still trips people up, since most tutorials predate it):
`size` → `vector_size`, `iter` → `epochs`, and **all similarity queries now go through `.wv`**
(`model.most_similar(...)` no longer works — it's `model.wv.most_similar(...)`).

```python
from gensim.models import Word2Vec

sentences = [
    ["natural", "language", "processing", "is", "fascinating"],
    ["word", "embeddings", "capture", "semantic", "meaning"],
    ["gensim", "makes", "training", "word", "vectors", "easy"],
    ["deep", "learning", "powers", "modern", "nlp", "systems"],
]

model = Word2Vec(
    sentences,
    vector_size=100,   # embedding dimensionality  (was 'size' in gensim 3.x)
    window=5,           # context window size
    min_count=1,         # ignore words with total frequency below this
    workers=4,            # parallel training threads
    sg=1,                  # 1 = skip-gram, 0 = CBOW (default)
    epochs=10,              # training iterations (was 'iter' in gensim 3.x)
)

print(model.wv.most_similar("nlp", topn=5))          # nearest neighbors
print(model.wv.similarity("nlp", "learning"))          # cosine similarity between two words
print(model.wv["nlp"])                                   # the raw 100-dim vector
print(model.wv.index_to_key[:10])                         # vocabulary, most frequent first
print(model.wv.key_to_index["nlp"])                         # word -> vocab index

model.save("word2vec.model")
loaded = Word2Vec.load("word2vec.model")

# --- Loading a HUGE pretrained model (e.g. Google's 300-dim, 3M-word News vectors) ---
import gensim.downloader as api
wv = api.load("word2vec-google-news-300")   # downloads ~1.6GB on first call, then cached
print(wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
# [('queen', 0.71), ('monarch', 0.62), ('princess', 0.60)]  -- the classic analogy example
```

📎 Docs: https://radimrehurek.com/gensim/models/word2vec.html

---

### 3.8 GloVe

**Global Vectors** — unlike Word2Vec's local context windows, GloVe trains on a global
word-word co-occurrence matrix for the whole corpus. In practice, GloVe vectors are usually
*downloaded pretrained* rather than trained from scratch.

```python
import gensim.downloader as api

glove = api.load("glove-wiki-gigaword-100")   # 100-dim GloVe vectors trained on Wikipedia + Gigaword
print(glove.most_similar("computer", topn=5))
print(glove.similarity("king", "queen"))

# --- Converting a raw GloVe .txt file (from https://nlp.stanford.edu/projects/glove/) to gensim format ---
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove2word2vec("glove.6B.100d.txt", "glove.6B.100d.word2vec.txt")
glove_vectors = KeyedVectors.load_word2vec_format("glove.6B.100d.word2vec.txt")
```

📎 Stanford GloVe project: https://nlp.stanford.edu/projects/glove/

---

### 3.9 fastText

Facebook's word-embedding approach that represents each word as a **bag of character n-grams**,
which means it can generate reasonable vectors for **out-of-vocabulary (OOV) words** and
misspellings — a key advantage over Word2Vec/GloVe for morphologically rich languages or noisy text.

```python
from gensim.models import FastText

sentences = [["natural", "language", "processing"], ["deep", "learning", "models"]]
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)

print(ft_model.wv.most_similar("processing"))
print(ft_model.wv["proccessing"])   # works even on a MISSPELLED, never-seen word — thanks to subwords!

# --- Facebook's own standalone fasttext library (faster, adds supervised text classification) ---
# pip install fasttext
import fasttext
# ft = fasttext.train_unsupervised("corpus.txt", model="skipgram")  # word vectors
# clf = fasttext.train_supervised("train.txt", label_prefix="__label__")  # text classifier
# clf.predict("this movie was fantastic")
```

---

### 3.10 ELMo (Contextual Embeddings, Legacy)

The first widely-used **contextual** embedding: unlike Word2Vec/GloVe, the vector for "bank"
differs depending on whether the sentence is about a river or a financial institution (via a
bidirectional LSTM language model). Historically important — it directly inspired BERT — but
**rarely used in new projects today**; transformer-based contextual embeddings (§3.11) have
superseded it almost everywhere.

```python
# Via TensorFlow Hub (the classic way ELMo was distributed):
# pip install tensorflow tensorflow-hub
import tensorflow_hub as hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")
embeddings = elmo.signatures["default"](tf.constant(["the quick brown fox"]))["elmo"]
# -> shape (1, num_tokens, 1024): one contextual 1024-dim vector per token
```

🔧 If you're starting a new project, skip ELMo — go straight to BERT-style embeddings (§3.11) or
sentence-transformers (§3.13), which are faster to use, better supported, and more accurate.

---

### 3.11 BERT Embeddings

BERT produces a **contextual** vector for every token — the same word gets a different embedding
depending on its sentence. You can pool these into a single document/sentence vector.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

text = "Natural language processing is fascinating."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state          # shape: (1, seq_len, 768)
cls_embedding = last_hidden_state[:, 0, :]               # [CLS] token — a common sentence-level summary
mean_pooled = last_hidden_state.mean(dim=1)                # mean-pooling over all tokens — often works better

print(cls_embedding.shape, mean_pooled.shape)   # torch.Size([1, 768]) torch.Size([1, 768])
```

🔧 For production-quality sentence/document embeddings, don't hand-roll BERT pooling like this —
use **sentence-transformers** (§3.13), which is specifically fine-tuned to produce meaningful
sentence-level cosine similarities (raw BERT [CLS]/mean-pooled vectors are known to cluster poorly).

---

### 3.12 Doc2Vec

Extends Word2Vec to learn embeddings for **entire documents**, not just words, by training a
unique "paragraph vector" alongside each document's word vectors.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [
    "Natural language processing enables computers to understand text",
    "Machine learning models learn patterns from data",
    "Deep learning uses neural networks with many layers",
]
tagged_docs = [TaggedDocument(words=doc.lower().split(), tags=[str(i)]) for i, doc in enumerate(documents)]

model = Doc2Vec(tagged_docs, vector_size=50, window=3, min_count=1, epochs=40, workers=4)

print(model.dv["0"])                                  # vector for document index 0 (note: .dv, not .wv)
print(model.dv.most_similar("0"))                        # most similar documents to document 0

# Infer a vector for a brand-new, unseen document
new_vector = model.infer_vector("computers can learn from text data".lower().split())
print(model.dv.most_similar([new_vector]))
```

---

### 3.13 Sentence Embeddings (Sentence-BERT)

**The modern standard** for sentence/document embeddings and semantic search. Sentence-BERT
(SBERT) fine-tunes BERT-style models with a siamese network structure so that cosine similarity
between sentence embeddings actually reflects semantic similarity — solving the exact weakness of
raw BERT pooling mentioned in §3.11.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # small, fast, 384-dim, great default

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)   # (3, 384)

# Current recommended API: model.similarity() computes a full pairwise similarity matrix
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# Semantic search: find the most similar sentence to a new query
query_embedding = model.encode("What's the weather like?")
scores = model.similarity(query_embedding, embeddings)[0]
best_match = sentences[scores.argmax()]
print(best_match)   # "The weather is lovely today."
```

📎 Docs: https://sbert.net — also see §6.3 for combining this with a vector database at scale.

---

### 3.14 RoBERTa & DistilBERT

Two of the most common BERT variants you'll encounter:

- **RoBERTa** ("Robustly Optimized BERT"): same architecture as BERT, but trained longer, on more
  data, with bigger batches, and *without* the Next Sentence Prediction objective — generally
  outperforms vanilla BERT on downstream tasks.
- **DistilBERT**: a *distilled* (compressed) version of BERT — ~40% smaller, ~60% faster, retains
  ~97% of BERT's performance. Use it when latency/cost matters more than squeezing out the last
  point of accuracy.

```python
from transformers import AutoTokenizer, AutoModel

# Drop-in replacements for "bert-base-uncased" — the AutoTokenizer/AutoModel API is identical
roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModel.from_pretrained("roberta-base")

distilbert_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")

# 🔧 Note: RoBERTa's tokenizer doesn't use [CLS]/[SEP] — it uses <s>/</s> instead.
# AutoTokenizer handles this automatically, but be aware if you're inspecting raw token IDs.
print(roberta_tok.tokenize("Hello world"))   # ['Hello', 'Ġworld']  -- 'Ġ' marks a preceding space (BPE)
```

---

### 3.15 Embedding Methods Comparison Table

| Method | Type | Dimensionality | Context-aware? | Handles OOV words? | Typical use case |
|---|---|---|---|---|---|
| One-Hot | Sparse | = vocab size | No | No | Toy examples, categorical features |
| Bag of Words | Sparse | = vocab size | No | No | Simple baselines, interpretable models |
| TF-IDF | Sparse | = vocab size | No | No | Search/IR, classification baselines |
| Word2Vec | Dense, static | 100-300 | No | No | Word similarity, feature input to classic ML |
| GloVe | Dense, static | 50-300 | No | No | Same as Word2Vec; strong pretrained options |
| fastText | Dense, static | 100-300 | No | **Yes** (subwords) | Noisy text, morphologically rich languages |
| Doc2Vec | Dense, static | 50-300 | Document-level only | No | Document similarity/clustering |
| ELMo | Dense, contextual | 1024 | **Yes** | Partial | Legacy — superseded by transformers |
| BERT (raw) | Dense, contextual | 768 (base) | **Yes** | **Yes** (subwords) | Fine-tuning for downstream tasks |
| Sentence-BERT | Dense, contextual | 384-1024 | **Yes** | **Yes** | Semantic search, clustering, retrieval (RAG) |


---

## Part 4 — Core NLP Tasks

### 4.1 Text Classification

**Classical approach** (fast, interpretable, great baseline — still very competitive for large
labeled datasets with clear keyword signals):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = ["I love this product", "Terrible experience", "Best purchase ever", "Worst service", "Highly recommend"]
labels = ["positive", "negative", "positive", "negative", "positive"]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression()),          # swap for LinearSVC() or MultinomialNB()
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

**Transformer approach** (state-of-the-art accuracy, especially with limited labeled data — via
transfer learning from a pretrained model):

```python
from transformers import pipeline

# Zero-shot: classify into ARBITRARY labels with no training data at all
zero_shot = pipeline("zero-shot-classification")
result = zero_shot(
    "I need to return this laptop, the screen arrived cracked.",
    candidate_labels=["billing", "technical support", "returns", "sales"],
)
print(result["labels"][0], result["scores"][0])   # 'returns', 0.87 (highest-scoring label)

# Fine-tuned: use an existing model trained on your exact task
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("This movie was a total waste of time."))
```

🔧 For training your *own* classifier on custom labels, see the full `Trainer`-based fine-tuning
walkthrough in §5.9 — and the complete worked example in Part 8.

---

### 4.2 Named Entity Recognition (NER)

Extracting real-world objects — people, organizations, locations, dates, money — from text.

```python
# --- Rule-based / classical: NLTK ---
import nltk
sentence = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
print(tree)   # a nltk.Tree with (PERSON ...), (ORGANIZATION ...), (GPE ...) subtrees

# --- spaCy: fast, accurate, production-ready ---
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)
for ent in doc.ents:
    print(ent.text, ent.label_, spacy.explain(ent.label_))
# Apple Inc. ORG 'Companies, agencies, institutions, etc.'
# Steve Jobs PERSON 'People, including fictional'
# Cupertino GPE 'Countries, cities, states'
# California GPE 'Countries, cities, states'

displacy.render(doc, style="ent", jupyter=True)   # highlighted-entity visualization

# --- Transformers: highest accuracy, especially for domain-specific fine-tuned models ---
from transformers import pipeline
ner = pipeline("ner", aggregation_strategy="simple")   # merges sub-word tokens into full entities
for ent in ner(sentence):
    print(ent["word"], ent["entity_group"], round(ent["score"], 3))
```

**Common NER entity labels (spaCy / OntoNotes 5 scheme):**

| Label | Meaning | Label | Meaning |
|---|---|---|---|
| PERSON | People, including fictional | DATE | Absolute or relative dates/periods |
| ORG | Companies, agencies, institutions | TIME | Times smaller than a day |
| GPE | Countries, cities, states | MONEY | Monetary values |
| LOC | Non-GPE locations (mountains, water) | PERCENT | Percentages |
| PRODUCT | Objects, vehicles, foods (not services) | CARDINAL | Numerals not covered elsewhere |
| EVENT | Named hurricanes, battles, wars, sports events | LAW | Named documents made into laws |
| NORP | Nationalities, religious/political groups | LANGUAGE | Any named language |

🔧 Need to train NER on **your own entity types** (e.g. drug names, part numbers)? spaCy supports
this directly: `python -m spacy init config` → edit `[components.ner]` → `spacy train`. See
https://spacy.io/usage/training for the full walkthrough.

---

### 4.3 Text Summarization

**Extractive** summarization selects and stitches together existing sentences; **abstractive**
summarization generates genuinely new sentences (like a human would).

⚠️ `gensim.summarization` (a TextRank-based extractive summarizer) was **removed in Gensim 4.0**
as an unmaintained module — don't rely on old tutorials that use `gensim.summarize()`. Use one of
the alternatives below instead.

```python
# --- Extractive: sumy (TextRank / LexRank / LSA-based sentence extraction) ---
# pip install sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

text = "Your long article text goes here. It can span many sentences and paragraphs..."
parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = TextRankSummarizer()
summary = summarizer(parser.document, sentences_count=3)
for sentence in summary:
    print(sentence)

# --- Extractive: pytextrank (TextRank as a spaCy pipeline component) ---
# pip install pytextrank
import spacy, pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
doc = nlp(text)
for sent in doc._.textrank.summary(limit_sentences=3):
    print(sent)
```

```python
# --- Abstractive (modern/recommended): an instruction-tuned model via the unified text-generation pipeline ---
# This approach works regardless of transformers version and is how summarization is
# actually done in production today — dedicated encoder-decoder models like BART/Pegasus
# still work great, but instruction-tuned causal LMs now match or beat them for most use cases.
from transformers import pipeline

summarizer = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")   # any instruct model works
messages = [{"role": "user", "content": f"Summarize the following text in 2-3 sentences:\n\n{text}"}]
output = summarizer(messages, max_new_tokens=120)
print(output[0]["generated_text"][-1]["content"])
```

```python
# --- Abstractive (classic dedicated seq2seq models — BART/Pegasus/T5) ---
# Still fully supported: the underlying model classes never went away, only the
# one-line pipeline("summarization", ...) SHORTCUT was retired in transformers v5 (see §5.9).
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
summary_ids = model.generate(**inputs, max_length=130, min_length=30, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

# On transformers <5.0, the one-liner shortcut also works:
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# summarizer(text, max_length=130, min_length=30, do_sample=False)
```

---

### 4.4 Sentiment Analysis

```python
# --- NLTK VADER: rule-based, tuned for SHORT/social text (tweets, reviews) — no training needed ---
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("This movie was absolutely fantastic! 😊"))
# {'neg': 0.0, 'neu': 0.406, 'pos': 0.594, 'compound': 0.8442}
# 'compound' is the normalized overall score: >0.05 positive, <-0.05 negative, else neutral

# --- TextBlob: quick and simple ---
from textblob import TextBlob
print(TextBlob("This movie was absolutely fantastic!").sentiment.polarity)   # 0.6 (range: -1 to 1)

# --- Transformers: highest accuracy, handles nuance/sarcasm far better ---
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("The plot was predictable, but the acting somehow saved the film."))

# For 5-star / fine-grained sentiment instead of binary:
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
print(classifier("It was okay, nothing special."))   # returns e.g. '3 stars'
```

🔧 **Rule-based (VADER) vs. transformer models:** VADER needs zero training/GPU and is genuinely
good on short, informal, emoji-heavy text (its lexicon was built from social media). For long-form
or nuanced text (sarcasm, mixed sentiment, domain-specific language), transformer models win by a
wide margin.

---

### 4.5 Machine Translation

```python
# --- Modern/recommended: instruction-tuned model via text-generation (version-agnostic) ---
from transformers import pipeline

translator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
messages = [{"role": "user", "content": "Translate to French. Reply with only the translation:\n\nHow are you today?"}]
output = translator(messages, max_new_tokens=50)
print(output[0]["generated_text"][-1]["content"])
```

```python
# --- Classic: dedicated MarianMT translation models (Helsinki-NLP/opus-mt-*) — one model per language pair ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-fr"    # English -> French; swap 'en-fr' for other pairs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer("How are you today?", return_tensors="pt")
translated = model.generate(**inputs)
print(tokenizer.decode(translated[0], skip_special_tokens=True))
# "Comment allez-vous aujourd'hui ?"

# On transformers <5.0, the one-liner shortcut also works:
# translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
# translator("How are you today?")
```

🔧 For dozens of language pairs from one single model, look at **NLLB** (`facebook/nllb-200-distilled-600M`)
or **M2M-100** — both are many-to-many multilingual translation models from Meta AI.

---

### 4.6 Question Answering

**Extractive QA** finds the answer as a literal span inside a given context passage.
**Generative QA** composes a free-form answer (what modern chat-style LLMs do by default).

```python
# --- Extractive QA: pull the exact answer span out of a context passage ---
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = "Where is Hugging Face based?"
context = "Hugging Face Inc. is a company based in New York City. It develops tools for building ML applications."

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

start = outputs.start_logits.argmax()
end = outputs.end_logits.argmax() + 1
answer = tokenizer.decode(inputs["input_ids"][0][start:end])
print(answer)   # "New York City"

# On transformers <5.0, the one-liner shortcut also works:
# qa = pipeline("question-answering")
# qa(question=question, context=context)
```

```python
# --- Generative/open-domain QA: ask an instruction-tuned model directly (works with or without context) ---
from transformers import pipeline

qa_chat = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely."}]
output = qa_chat(messages, max_new_tokens=30)
print(output[0]["generated_text"][-1]["content"])
```

🔧 For QA over **your own large document collections** (not a single short passage), what you
actually want is Retrieval-Augmented Generation — see §6.2.

