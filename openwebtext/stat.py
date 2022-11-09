import os
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np


def filter_not(line: str) -> bool:
    # words = word_tokenize(line)
    words = line.split()
    return "not" in words or "n't" in words


if __name__ == '__main__':
    random.seed(1234)
    filenames = sorted(os.listdir("openwebtext"))
    sampled_filenames = random.sample(filenames, 100)
    print("Sampled filenames:", sampled_filenames)

    all_lines = []
    for filename in tqdm(sampled_filenames):
        if filename.endswith("data"):
            with open(f"openwebtext/{filename}", "r") as f:
                lines = [line.strip() for line in f.readlines() if len(line.strip()) != 0]
                all_lines += lines

    print(f"Total number of lines: {len(all_lines)}")
    print(f"Total number of lines containing 'not': {len([line for line in all_lines if filter_not(line)])}")
    
    word_freq = Counter([word for line in all_lines for word in line.split()]).most_common(50000)
    print("Most common words:")
    for word, freq in word_freq[:50]:
        print(word, freq)
    vocab = {word: i for i, (word, freq) in enumerate(word_freq)}

    words_after_is, words_after_is_not = [0] * (len(vocab) + 1), [0] * (len(vocab) + 1)
    for line in all_lines:
        words = line.split()
        for i in range(len(words) - 1):
            if words[i] == "is" and words[i + 1] == "not":
                if i + 2 < len(words):
                    words_after_is_not[vocab.get(words[i + 2], -1)] += 1
            elif words[i] == "is":
                words_after_is[vocab.get(words[i + 1], -1)] += 1
    
    normalized_words_after_is = [freq / sum(words_after_is) for freq in words_after_is]
    normalized_words_after_is_not = [freq / sum(words_after_is_not) for freq in words_after_is_not]
    print("Most common words after 'is':")
    for word, freq in sorted(zip(vocab.keys(), normalized_words_after_is), key=lambda x: x[1], reverse=True)[:50]:
        print(word, freq)
    print("Most common words after 'is not':")
    for word, freq in sorted(zip(vocab.keys(), normalized_words_after_is_not), key=lambda x: x[1], reverse=True)[:50]:
        print(word, freq)


    np_normalized_words_after_is = np.array(normalized_words_after_is)
    np_normalized_words_after_is_not = np.array(normalized_words_after_is_not)
    print(np_normalized_words_after_is[:50], np_normalized_words_after_is[:50].sum())
    print(np_normalized_words_after_is_not[:50], np_normalized_words_after_is_not[:50].sum())
    
    # KL-divergence
    np_normalized_words_after_is += 1e-10
    np_normalized_words_after_is_not += 1e-10
    print("KL-divergence-1:", np.sum(np_normalized_words_after_is_not * np.log(np_normalized_words_after_is_not / np_normalized_words_after_is)))
    print("KL-divergence-2:", np.sum(np_normalized_words_after_is * np.log(np_normalized_words_after_is / np_normalized_words_after_is_not)))
    