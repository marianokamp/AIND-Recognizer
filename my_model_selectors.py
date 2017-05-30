import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def pick_best(self, results):
        print("results", results)
        assert(len(results) > 0)
        best_n = sorted(results, key=lambda t: -t[1])[0][0] # best result, n column
        print("best_n", best_n)
        return best_n

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        results = [] # (n, score)
       
        for n in range(self.min_n_components, self.max_n_components+1):
       
            print("n=", n, "X", len(self.X), "lengths=", self.lengths)
            logL = self.base_model(n).score(self.X, self.lengths)
            dimension_count = self.X.shape[0]
            parameter_count = n ** 2 + (dimension_count * 2 * n) -1
            N = len(self.X)
            #print("dimension_count", dimension_count)
            #print("parameter_count", parameter_count)

            bic = -2 * logL + parameter_count * np.log(N)
            print("bic, logL, n, N, parameter_count", bic, logL, n, N, parameter_count)
            results.append((n, bic))     

        return self.base_model(self.pick_best(results))

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
   
        # main objective is to find out how different values of n, number of
        # states, affect the logL.
        # Will iterate over all ns and then compare the logL for this specific
        # word with this specific n to the logL of all other words with the
        # same n.
        # The n which yields the highest logL will be returned.

        results = [] # (n, score)        
        print(self.this_word)

        for n in range(self.min_n_components, self.max_n_components+1):
            
            model = self.base_model(n)

            this_word_score    = None
            other_words_scores = [] 

            for word in self.words:

                try:
                    X_train, lengths_train = self.hwords[word]

                    score = model.score(X_train, lengths_train)
                    
                    if word == self.this_word:
                        assert(not this_word_score)
                        this_word_score = score
                    else:
                        other_words_scores.append(score)
                except:
                    #print("Oops")
                    pass
            
            if this_word_score or other_words_scores:
                mean_other_words_scores = np.mean(other_words_scores)
                results.append((n, max(mean_other_words_scores, this_word_score)))
                print(mean_other_words_scores, this_word_score)

        return self.base_model(self.pick_best(results))

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        features = ['polar-lr-norm', 'polar-rr-norm', 'polar-ltheta-norm', 'polar-rtheta-norm']
        split_method = KFold(n_splits=min(2, len(self.lengths)))
        
        results = [] # (n, score)
        
        for n in range(self.min_n_components, self.max_n_components+1):
            #print("self.sequences=", len(self.sequences))
            #print ("n={}".format(n))
            
            cv_scores = []

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
            
                    model = self.base_model(n).fit(X_train, lengths_train)
                    score = model.score(X_test, lengths_test)
                    cv_scores.append(score)
                    
                except:
                    pass
                    #print("oops")
            results.append((n, np.mean(cv_scores))) 
        print("results", results)
        
        best_score = -math.inf
        best_n = -math.inf

        for result in results:
            if result[1] > best_score:
                best_n, best_score = result
        print("best_n", best_n, "best_score", best_score)

        return self.base_model(best_n)

if __name__ == '__main__':
    import unittest
    from asl_test_model_selectors import TestSelectors
    suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
    unittest.TextTestRunner().run(suite)
