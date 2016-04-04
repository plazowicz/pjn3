# coding=utf-8
from utils import ProbabilityOfErrorGivenForm, chunks
import codecs
import cPickle
import pickle
from multiprocessing import Pool
from Levenshtein import editops
from utils import df
import numpy as np
import operator
from progressbar import ProgressBar, Percentage, ETA
import h5py

__author__ = 'mateuszopala'

with codecs.open('data/priors.pkl', 'rb') as f:
    priors = cPickle.load(f)

with h5py.File('data/p_of_w_given_c_model.hdf5', 'r') as f:
    insertion_cm = f['i'][:]
    deletion_cm = f['d'][:]
    substitution_cm = f['s'][:]

with open('data/counts.pkl', 'r') as f:
    counts_dict = pickle.load(f)
with open('data/char_to_index.pkl', 'r') as f:
    char_to_index = pickle.load(f)


def p_of_w_given_c(w, c):
    prob = 0.
    w = u'$' + w
    candidate_correction = u'$' + c
    ops = editops(w, candidate_correction)
    for op, spos, dpos in ops:
        nominator, denominator = None, None
        if op == "insert":
            c2 = candidate_correction[dpos]
            c1 = w[spos - 1]
            nominator = insertion_cm[char_to_index[c1], char_to_index[c2]]
            denominator = float(counts_dict[c1])
        if op == "delete":
            c2 = w[spos]
            c1 = w[spos - 1]
            nominator = deletion_cm[char_to_index[c1], char_to_index[c2]]
            denominator = float(counts_dict[c1 + c2])
        if op == "replace":
            c2 = candidate_correction[dpos]
            c1 = w[spos]
            nominator = substitution_cm[char_to_index[c1], char_to_index[c2]]
            denominator = float(counts_dict[c1])
        if nominator is not None and denominator is not None:
            prob += np.log(nominator / denominator)
    return prob


WORD = None


def get_most_probable_from_chunk(forms):
    global WORD
    best_form = None
    best_prob = float('-inf')
    widgets = ['Computing most probable from chunk: ', Percentage(), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=len(forms)).start()
    for i, c in enumerate(forms):
        prob = p_of_w_given_c(WORD, c) + priors[c]
        # if c==u'dupa':
        #     print prob
        #     print best_prob
        if prob > best_prob:
            best_form = c
            best_prob = prob
        pbar.update(i + 1)
    pbar.finish()
    return best_form, best_prob


class NaiveBayesSpellChecker(object):
    def __init__(self, n_jobs=4):
        self.p_of_w_given_c = ProbabilityOfErrorGivenForm()
        with codecs.open("data/formy_utf8.txt", "r", "utf-8") as f:
            self.forms = f.read().splitlines()
        self.n_jobs = n_jobs

    def correct(self, word):
        global WORD
        WORD = word
        p = Pool(self.n_jobs)
        chunk_size = int(len(self.forms) / self.n_jobs)
        # form, max_prob = get_most_probable_from_chunk(self.forms)
        arguments = chunks(self.forms, chunk_size)
        results = p.map(get_most_probable_from_chunk, arguments)
        p.close()
        p.join()
        form, max_prob = max(results, key=operator.itemgetter(1))
        return form


if __name__ == "__main__":
    nb_sc = NaiveBayesSpellChecker()
    print nb_sc.correct(u'dópa')

    with codecs.open("data/bledy.txt", "r", "utf-8") as f:
        errors, corrections = map(lambda x: list(x), zip(*[line.split(';') for line in f.read().splitlines()]))

    count = 0.
    for error, correction in zip(errors, corrections):
        pred = nb_sc.correct(error)
        if pred == correction:
            count += 1

    print "accuracy: %f" % (count / len(corrections))
