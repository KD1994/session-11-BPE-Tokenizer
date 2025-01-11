import os
import sys
import glob
import regex as re
import pandas as pd
import requests
import unicodedata
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
from tqdm import tqdm


class GujaratiBPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.compression_ratio = 0.
        self.merges = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        # applies on the entire corpus
        self.global_pattern = re.compile(r""" [\p{L}\p{M}\p{N}]+|[\p{L}\p{M}\p{N}]+|[^\r\n\p{L}\p{M}\p{N}]+""")
        # applies on each words to separate morphpligical transformation ending with "ન" or "મ"
        self.local_pattern = re.compile(r"""([\s\p{L}\p{M}]+|[\s\p{L}\p{M}\p{N}]+)([નમ](?:\p{M}))$""")
        self.eng2guj = self.get_eng_to_guj_digits_mapping()
        self.guj_unicode_df = self.get_guj_unicodes()
        # Initialize basic Odia character vocabulary
        self.base_vocab = set()
        # Add basic Odia characters (vowels, consonants, marks)
        self._initialize_base_vocab()


    def get_guj_unicodes(self):
        res = requests.get("https://www.unicode.org/Public/UNIDATA/UnicodeData.txt")
        lines = res.text.splitlines()
        lines = [",".join(line.split(";")[:2]) for line in lines if "GUJARATI" in line]
        data = {
            "code": [l.split(",")[0] for l in lines],
            "name": [l.split(",")[-1] for l in lines],
            "char": [unicodedata.lookup(l.split(",")[1]) for l in lines],
        }
        df = pd.DataFrame(data)
        return df
    

    def _initialize_base_vocab(self):
        """Initialize vocabulary with basic Odia characters"""
        # Vowels
        self.base_vocab.update(self.guj_unicode_df["char"].to_list())
        # Whitespace characters with period.
        self.base_vocab.update([' ', '\n', '\t', "."])


    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs in the vocabulary"""
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i + 2])] += 1
        return pairs


    def _merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of the most frequent pair"""
        first, second = pair
        new_words = []
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        
        return new_words

    def get_eng_to_guj_digits_mapping(self):
        e2g = dict()
        # Add digits 0 to 9
        for i in range(10):
            e2g[str(i)] = unicodedata.lookup(f"GUJARATI DIGIT {unicodedata.name(chr(48+i)).split()[-1]}")
        
        return e2g


    def remove_eng_words(self, text):
        pat = re.compile(r"[a-zA-Z]+", re.IGNORECASE)
        text = " ".join(re.sub(pat, "", text).split())
        # text = re.sub(pat, "", text))
        return text


    def eng_to_guj_digits(self, text, e2g):
        new_text = ""
        for ch in text:
            if ch.isdigit() and ch not in e2g.values():
                new_text += e2g[ch]
            else:
                new_text += ch

        return new_text
    
    
    def process_text_with_regex(self, text):
        split_text = re.findall(self.global_pattern, text)
        new_text =[]
        for t in split_text:
            split_words = re.findall(self.local_pattern, t)
            # print(f"word: {t} --> word split: {split_words}")
            if split_words:
                for item in split_words:
                    if isinstance(item, tuple):
                        w = [i for i in item if i != ""]
                        # print(f"item: {item} --> {w}")
                        new_text.extend(w)
            else:
                new_text.append(t)

        return new_text
    
    def tokenize_text(self, texts: List[str]):
        """
        Takes a list of text and provides list of processed words required for the encoding.

        Args:
            texts (List[str]): text lines

        Returns:
            list: list of extraced words from the text lines
        """
        processed_text = []
        for t in tqdm(texts, desc="preprocessing", colour="green", bar_format="{l_bar}{bar:30}{r_bar}"):
            processed_text.append(self.eng_to_guj_digits(self.remove_eng_words(t), self.eng2guj))

        processed_text = " ".join(processed_text)
        words = self.process_text_with_regex(processed_text)

        return words
    

    def train(self, texts: List[str], min_freq: int = 2) -> None:
        """Train BPE model on texts"""
        
        tokens = self.tokenize_text(texts)
        words = tokens
                    
        vocab = self.base_vocab.copy()
        num_merges = self.vocab_size - len(self.special_tokens) - len(vocab)
        # print("num_merges : ", num_merges)
        # Perform BPE merges
        train_bar = tqdm(range(num_merges),
                         desc="Merging pairs",
                         total=num_merges, 
                         colour="blue", 
                         file=sys.stdout, 
                         bar_format="{l_bar}{bar:30}{r_bar}"
        )
        for i in train_bar:
            pairs = self._get_stats(words)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])
            if best_pair[1] < min_freq:
                break

            pair = best_pair[0]
            new_token = ''.join(pair)
            vocab.add(new_token)
            #print("merging ..", pair)
            # print(len(vocab))
            # Record the merge operation
            self.merges[pair] = new_token
            
            # Merge the pair in all words
            words = self._merge_vocab(words, pair)

        # Build final vocabulary
        self.vocab = {**self.special_tokens}
        idx = len(self.special_tokens)
        for token in sorted(vocab):
            self.vocab[token] = idx
            idx += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.compression_ratio = len(tokens) / len(words)
        print("tokens length:", len(tokens))
        print("tokens length after merge operation:", len(words))
        print(f"compression ratio: {len(tokens) / len(words):.2f}X")


    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges"""

        # odia_word_pattern = re.compile(r""" ?[\u0B00-\u0B7F]+| ?[^\s]+|\s+(?!\S)|\s+""")
        # extracted_words = odia_word_pattern.findall(text)

        # words = [list(word) for word in extracted_words]
        #words = [list(text)]

        tokenized_words = self.tokenize_text([text])
        words = [list(word) for word in tokenized_words]
        # print("Before merges: ", words)
        
        # Apply merges in order
        for pair, merged in self.merges.items():
            words = self._merge_vocab(words, pair)
        # print("After mergers: ", words)

        # Convert to token IDs
        result = []
        for word in words:
            for token in word:
                if token in self.vocab.keys():
                    result.append(self.vocab[token])
                else:
                    result.append(self.special_tokens['<UNK>'])
        
        return result


    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return ''.join(self.inverse_vocab.get(id, '<UNK>') for id in ids)


    def calculate_compression_ratio(self, text: str) -> float:
        """Calculate compression ratio"""
        encoded = self.encode(text)
        return len(text) / len(encoded)


    def save(self, path: str) -> None:
        """Save tokenizer state"""
        # Convert tuple keys to strings for JSON serialization
        serializable_merges = {f"{first}|{second}": merged 
                              for (first, second), merged in self.merges.items()}
        
        data = {
            'vocab': self.vocab,
            'merges': serializable_merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'compression_ratio': self.compression_ratio
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    @classmethod
    def load(cls, path: str) -> 'GujaratiBPETokenizer':
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        
        # Convert string keys back to tuples
        tokenizer.merges = {tuple(k.split('|')): v 
                           for k, v in data['merges'].items()}
        
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.compression_ratio = data['compression_ratio']
        print(f"Tokenizer loaded!")
        return tokenizer


# if __name__ == "__main__":
#     # train
#     data_path = os.path.join("data")
#     news_articles = glob.glob(os.path.join(data_path, "news dataset", "*.txt"))
#     cc100_dataset = glob.glob(os.path.join(data_path, "cc100-Gujarati", "*.txt"))
#     indic_dataset = glob.glob(os.path.join(data_path, "IndicCorp", "*.txt"))
#     final_dataset = news_articles + cc100_dataset + indic_dataset

#     texts = []
#     c = 0
#     for article in final_dataset:
#         with open(os.path.join(article), "r", encoding='utf-8') as f:
#             texts.append(f.readline().strip())

#     tokenizer = GujaratiBPETokenizer()
#     tokenizer.train(texts)
#     tokenizer.save(os.path.join("Gujarati_tokenizer.json"))

#     # test
#     tokenizer = GujaratiBPETokenizer().load("Gujarati_tokenizer.json")
#     text1 = "ચામરાજનગર ભારત દેશના દક્ષિણ ભાગમાં આવેલા કર્ણાટક રાજ્યના ચામરાજનગર જિલ્લામાં આવેલું એક નગર છે. ચામરાજનગરમાં ચામરાજનગર જિલ્લાનું મુખ્યાલય છે."
#     enc_text1 = tokenizer.encode(text1)
#     print(enc_text1, len(enc_text1))
#     text2 = tokenizer.decode(enc_text1)
#     print(text2)

#     assert text1 == text2, "Problem with BPE!!"
