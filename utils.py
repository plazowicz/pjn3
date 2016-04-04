# coding=utf-8
import codecs
import pickle
import numpy as np
import os
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import editops
from progressbar import ProgressBar, ETA, Percentage
import h5py
import cPickle

__author__ = 'mateuszopala'


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def df():
    return 1


class ProbabilityOfErrorGivenForm(object):
    def __init__(self, dump_path=None):
        with codecs.open("data/bledy.txt", "r", "utf-8") as f:
            errors, corrections = map(lambda x: list(x), zip(*[line.split(';') for line in f.read().splitlines()]))
        with open("data/alphabet.pkl", 'r') as f:
            self.alphabet = list(pickle.load(f)) + [u'$']
        n = len(self.alphabet)
        self.char_to_index = {c: i for i, c in enumerate(self.alphabet)}

        self.deletion_cm = np.ones((n, n))
        self.insertion_cm = np.ones((n, n))
        self.substitution_cm = np.ones((n, n))

        self.counts_dict = defaultdict(df)
        model = CountVectorizer(analyzer='char', ngram_range=(1, 2))

        errors = [u'$' + error for error in errors]
        corrections = [u'$' + correction for correction in corrections]

        counts_dense = model.fit_transform(errors + corrections).todense().sum(axis=0)
        for i, feature_name in enumerate(model.get_feature_names()):
            self.counts_dict[feature_name] += counts_dense[0, i]

        for error, correction in zip(errors, corrections):
            ops = editops(error, correction)
            for op, spos, dpos in ops:
                if op == 'insert':
                    c2 = correction[dpos]
                    c1 = error[spos - 1]
                    self.insertion_cm[self.char_to_index[c1], self.char_to_index[c2]] += 1
                if op == 'delete':
                    c2 = error[spos]
                    c1 = error[spos - 1]
                    self.deletion_cm[self.char_to_index[c1], self.char_to_index[c2]] += 1
                if op == 'replace':
                    c2 = correction[dpos]
                    c1 = error[spos]
                    self.substitution_cm[self.char_to_index[c1], self.char_to_index[c2]] += 1
        if dump_path is not None:
            with h5py.File(dump_path, 'w') as f:
                f.create_dataset('i', data=self.insertion_cm)
                f.create_dataset('d', data=self.deletion_cm)
                f.create_dataset('s', data=self.substitution_cm)
            with open('data/counts.pkl', 'w') as f:
                pickle.dump(self.counts_dict, f)
            with open('data/char_to_index.pkl', 'w') as f:
                pickle.dump(self.char_to_index, f)

    def __call__(self, word, candidate_correction):
        prob = 0.
        word = u'$' + word
        candidate_correction = u'$' + candidate_correction
        ops = editops(word, candidate_correction)
        for op, spos, dpos in ops:
            nominator, denominator = None, None
            if op == "insert":
                c2 = candidate_correction[dpos]
                c1 = word[spos - 1]
                nominator = self.insertion_cm[self.char_to_index[c1], self.char_to_index[c2]]
                denominator = float(self.counts_dict[c1])
            if op == "delete":
                c2 = word[spos]
                c1 = word[spos - 1]
                nominator = self.deletion_cm[self.char_to_index[c1], self.char_to_index[c2]]
                denominator = float(self.counts_dict[c1 + c2])
            if op == "replace":
                c2 = candidate_correction[dpos]
                c1 = word[spos]
                nominator = self.substitution_cm[self.char_to_index[c1], self.char_to_index[c2]]
                denominator = float(self.counts_dict[c1])
            if nominator is not None and denominator is not None:
                prob += np.log(nominator / denominator)
        return prob


class PriorsCalculator(object):
    def __init__(self):
        with codecs.open("data/formy_utf8.txt", "r", "utf-8") as f:
            self.forms = f.read().splitlines()

    def calculate(self, corpora):
        model = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        stats = model.fit_transform([corpora])
        prior_per_form = {}
        widgets = ['Computing priors for forms: ', Percentage(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(self.forms)).start()
        for i, form in enumerate(self.forms):
            try:
                form_index = model.vocabulary_[form]
                prior_per_form[form] = stats[0, form_index]
            except KeyError:
                prior_per_form[form] = 0
            pbar.update(i + 1)
        pbar.finish()
        n = sum(prior_per_form.values())
        prior_per_form = {form: np.log((prior + 1.) / (n + len(self.forms))) for form, prior in
                          prior_per_form.iteritems()}
        return prior_per_form


def compute_priors(text_files=['big.txt'], dump=False):
    corporas = []
    # for fn in ['popul.iso.utf8', 'proza.iso.utf8', 'publ.iso.utf8', 'wp.iso.utf8', 'dramat.iso.utf8']:
    for fn in text_files:
        path = os.path.join('data', fn)
        with codecs.open(path, 'r', 'utf-8') as f:
            corpora = f.read()

    calc = PriorsCalculator()

    prior_per_form = calc.calculate(corpora)
    if dump:
        with open('data/priors.pkl', 'wb') as f:
            cPickle.dump(prior_per_form, f)


if __name__ == "__main__":
    # prob_of_error_given_form = ProbabilityOfErrorGivenForm(dump_path='data/p_of_w_given_c_model.hdf5')
    compute_priors(dump=True)