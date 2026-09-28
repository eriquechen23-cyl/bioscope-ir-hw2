"""Independent implementation of Porter's 1980 suffix-stripping rules.

Reference: https://tartarus.org/martin/PorterStemmer/def.txt
Original algorithm, including ABLI -> ABLE (not NLTK extensions).
Nonalphabetic biomedical terms such as GLP-1 are deliberately preserved.
"""
from functools import lru_cache


def consonants(word: str) -> list[bool]:
    result = []
    for c in word:
        result.append(c not in "aeiou" if c != "y" else not result[-1] if result else True)
    return result


def measure(word: str) -> int:
    flags = consonants(word)
    return sum(not a and b for a, b in zip(flags, flags[1:]))


def has_vowel(word: str) -> bool:
    return any(not c for c in consonants(word))


def cvc(word: str) -> bool:
    return len(word) >= 3 and consonants(word)[-3:] == [True, False, True] and word[-1] not in "wxy"


def replace_suffix(word: str, rules: dict[str, str], minimum: int) -> str:
    for suffix in sorted(rules, key=len, reverse=True):
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            return stem + rules[suffix] if measure(stem) > minimum else word
    return word


@lru_cache(maxsize=50000)
def stem(word: str) -> str:
    word = word.lower()
    if not word.isascii() or not word.isalpha():
        return word
    # 1a: longest suffix wins, including when a condition fails.
    word = replace_suffix(word, {"sses": "ss", "ies": "i", "ss": "ss", "s": ""}, -1)
    # 1b.
    if word.endswith("eed"):
        if measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and has_vowel(word[:-len(suffix)]):
                word = word[:-len(suffix)]
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif len(word) >= 2 and word[-1] == word[-2] and consonants(word)[-1] and word[-1] not in "lsz":
                    word = word[:-1]
                elif measure(word) == 1 and cvc(word):
                    word += "e"
                break
    # 1c.
    if word.endswith("y") and has_vowel(word[:-1]):
        word = word[:-1] + "i"
    # 2 and 3.
    word = replace_suffix(word, {
        "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize",
        "abli": "able", "alli": "al", "entli": "ent", "eli": "e", "ousli": "ous", "ization": "ize",
        "ation": "ate", "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
        "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
    }, 0)
    word = replace_suffix(word, {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": ""}, 0)
    # 4: ION additionally requires a stem ending in S or T.
    for suffix in sorted(("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
                          "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"), key=len, reverse=True):
        if word.endswith(suffix):
            base = word[:-len(suffix)]
            if measure(base) > 1 and (suffix != "ion" or base.endswith(("s", "t"))):
                word = base
            break
    # 5a and 5b.
    if word.endswith("e"):
        base = word[:-1]
        if measure(base) > 1 or (measure(base) == 1 and not cvc(base)):
            word = base
    if word.endswith("ll") and measure(word) > 1:
        word = word[:-1]
    return word
