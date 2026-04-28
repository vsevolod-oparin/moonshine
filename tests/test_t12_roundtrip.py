"""T12: Tokenizer roundtrip test — encode → decode 1000 Russian sentences.

Tests: normal text, numbers, hyphenated words, abbreviations, dates, English loanwords.
"""

import sentencepiece as spm
import random
import re
import sys
from pathlib import Path

DATA_DIR = Path("data")
random.seed(42)


def load_test_sentences():
    sentences = []

    with open(DATA_DIR / "tokenizer_corpus.txt", "r") as f:
        corpus = [l.strip() for l in f if l.strip()]

    random.shuffle(corpus)
    sentences.extend(corpus[:600])

    numbers = [
        "двадцать три", "сто пятьдесят", "тысяча девятьсот восемьдесят пятый",
        "двести сорок один", "ноль целых пять десятых", "три тысячи двадцать",
        "пятнадцать рублей", "семьсот двадцать три", "сто сорок четыре",
        "девяносто девять", "одиннадцать", "двести восемь",
    ] * 17
    sentences.extend(numbers[:200])

    specials = [
        "сша", "мгу", "оон", "рф", "фсб", "мид", "цру", "нato",
        "кто-то", "где-нибудь", "какой-либо", "по-русски", "по-другому",
        "жизнь-то", "человек-то", "давным-давно",
        "1 мая", "8 марта", "23 февраля", "9 мая", "31 декабря",
        "иванов-иванов", "смирнов-соколов",
        "бизнес", "маркетинг", "офис", "менеджер", "браузер", "сайт",
        "стартап", "дедлайн", "логин", "клик", "кэш", "хостинг",
        "интернет", "компьютер", "программа", "дизайн", "проект",
        "фильм", "шоу", "хит", "тренд", "блог",
    ]
    sentences.extend(specials * 15)

    return sentences


def check_roundtrip(sp, sentences, label):
    passed = 0
    failed = 0
    failures = []

    for s in sentences:
        ids = sp.encode(s, out_type=int)
        decoded = sp.decode(ids)
        if decoded == s:
            passed += 1
        else:
            failed += 1
            if len(failures) < 10:
                failures.append((s, decoded))

    total = passed + failed
    print(f"\n{label}: {passed}/{total} roundtrip correctly ({failed} failures)")
    if failures:
        print("  Sample failures:")
        for orig, dec in failures:
            print(f"    '{orig}' -> '{dec}'")

    return failed == 0


def main():
    sentences = load_test_sentences()
    print(f"Test sentences: {len(sentences)}")

    all_pass = True
    for vocab_size in [256, 512, 1024]:
        model_path = str(DATA_DIR / f"tokenizer_{vocab_size}.model")
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        ok = check_roundtrip(sp, sentences, f"vocab={vocab_size}")
        all_pass = all_pass and ok

    if all_pass:
        print("\nT12: PASS — all roundtrips correct")
    else:
        print("\nT12: FAIL — some roundtrips failed")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
