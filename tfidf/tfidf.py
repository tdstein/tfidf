__author__ = 'tdstein'

import math

class TFIDF(object):

    def __init__(self, documents):
        self.documents = [document.lower() for document in documents]

    def evaluate(self):
        idf = self.idf()
        for document in self.documents:
            tfidf_scores = {}
            tf = self.tf(document)
            for term in document.split():
                tfidf_scores[term] = tf[term] * idf[term]
            print tfidf_scores



    def tf(self, document):
        terms = document.split()
        term_counts = {}
        for term  in terms:
            if term not in term_counts:
                term_counts[term] = 1
            else:
                term_counts[term] += 1

        return term_counts

    def idf(self):
        term_counts = {}
        for document in self.documents:
            for term in set(document.split()):
                if term not in term_counts:
                    term_counts[term] = 1
                else:
                    term_counts[term] += 1

        num_of_documents = len(self.documents)
        for k, v in term_counts.iteritems():
            term_counts[k] = 1 + math.log10((num_of_documents / v))

        return term_counts