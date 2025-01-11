<h1 align="center">
  <b>BPE Tokenizer trained on Gujarati Language</b>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-31012/">
    <img src="https://img.shields.io/badge/Python-3.10.12-blue" alt="Python 3.10.12">
  </a>
  <a href="https://huggingface.co/spaces/tranquilkd/GujaratiTokenizer">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"/>
  </a>
</p>

<p align="center">
Train a BPE tokenizer on Gujarati language with 5000+ tokens and compression ratio of more than 3
</p>

# DATASET

3 different datasets are used for making the corpus:

1. Gujarati News Articles
2. AI4Bharat-IndicNLP Corpus
3. cc100-gujarati

# DATA PROCESSING

1. Remove English alphabets
2. Replace English digits with Gujarati digits
3. Apply custom RegEx
  - applies on the entire corpus
    ```global_pattern = re.compile(r""" [\p{L}\p{M}\p{N}]+|[\p{L}\p{M}\p{N}]+|[^\r\n\p{L}\p{M}\p{N}]+""")```
  - applies on each words to separate morphpligical transformation ending with "ન" or "મ"
    ```local_pattern = re.compile(r"""([\s\p{L}\p{M}]+|[\s\p{L}\p{M}\p{N}]+)([નમ](?:\p{M}))$""")```

# TRAINIG STATS

| Metric                        | Value     |
|-------------------------------|-----------|
| Tokens length                 | 2304322   |
| Tokens length after merging   | 241668    |
| Compression ratio             | 9.54X     |


# CITETION

```bibtex
link: https://data.mendeley.com/datasets/r46cwx8xw6/1

Patel, Jainish (2024), “Gujarati News Articles”, Mendeley Data, V1, doi: 10.17632/r46cwx8xw6.1

@article{kunchukuttan2020indicnlpcorpus,
    title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085},
}

https://metatext.io/redirect/cc100-gujarati
```
