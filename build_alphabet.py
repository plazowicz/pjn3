__author__ = 'mateuszopala'
import codecs
import pickle


if __name__ == "__main__":
    with codecs.open('data/bledy.txt', 'r', 'utf-8') as f:
        r, t = zip(*[line.split(';') for line in f.read().splitlines()])
        chars = ''.join(r) + ''.join(t)
    with codecs.open('data/formy_utf8.txt', 'r', 'utf-8') as f:
        chars2 = ''.join(f.read().splitlines())
    alphabet = list(set(chars + chars2))
    with open('data/alphabet.pkl', 'w') as f:
        pickle.dump(alphabet, f)
