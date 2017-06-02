import warnings
from asl_data import SinglesData
import math

#from asl_test_recognizer import TestRecognize

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    for test_idx in range(test_set.num_items):
        X_test, lengths_test = test_set.get_item_Xlengths(test_idx)
        
        scores = {} # word: logL
        best_score = -math.inf
        best_guess = None

        for word, model in models.items():
            #print("recognize word:", word, "model:", model)
            try:
                score = -math.inf
                score = model.score(X_test, lengths_test)
                #print("score", score)
                scores[word] = score

            except:
                scores[word] = -math.inf

            if not best_guess or score > best_score:
                best_score = score
                best_guess = word
      
        probabilities.append(scores)
        guesses.append(best_guess)
    
    return probabilities, guesses
