__author__ = 'tdstein'

import math
import re
import operator
from collections import OrderedDict
from stemming.porter2 import stem

"""
Term Frequency - Inverse Document Frequency, or TFIDF, is a method of determining the likely impact a term has towards the
distinction of a phrase in relationship to a other phrases in a corpus of documents. Once an estimated impact of each
term in a corpus of documents has been found, cosine similarity can be used to find similar documents.
"""

class TFIDF(object):

    documents = []

    # A static set of IDF score by token.
    __idf_by_term = {}
    __tf_by_term_by_document = {}

    def __init__(self, documents):
        self.documents += documents

    def add_document(self, document):
        """
        Add a single document to the training set
        """
        self.documents.append(document)

        # Clear out the idf score for each term in this new document. We must recompute it.
        for term in document.text:
            if term in self.__idf_by_term:
                self.__idf_by_term.pop(term, None)

    def add_documents(self, documents):
        """
        Add a set of documents to the training set.
        """
        for document in documents:
            self.add_document(document)

    def evaluate(self):
        """
        Finds similar documents, sorted by likelihood of similarity, for every supplied document.

        The similarity score is found the cosine similarity method.

        :return: A dictionary of every document mapped to a list of similar documents ordered by most similar.
        """

        # Calculate the Term Frequency - Inverse Document Frequency of each term in the corpus
        tfidf_by_term_by_document = {}
        for document in self.documents:
            tfidf_by_term = {}
            for term in document.text.split():
                tf = self.__tf__(term, document)
                idf = self.__idf__(term)
                tfidf_by_term[term] = tf * idf
            tfidf_by_term_by_document[document] = OrderedDict(sorted(tfidf_by_term.items()))

        # Find the nearest neighbors by document
        neighbors_by_document = {}
        for target in self.documents:
            neighbors = {}
            for neighbor in self.documents:
                if neighbor != target:
                    # Find the common terms in both the target and the neighbor
                    common_terms = set(target.text.split()).intersection(set(neighbor.text.split()))

                    tvector = [tfidf for term, tfidf in tfidf_by_term_by_document[target].iteritems() if term in common_terms]
                    nvector = [tfidf for term, tfidf in tfidf_by_term_by_document[neighbor].iteritems() if term in common_terms]

                    # Assert neither vector is empty
                    if tvector and nvector:
                        similarity = self.__cosine_distance__(tvector, nvector)
                        if similarity > 0:
                          neighbors[neighbor] = similarity

            # Sort the neighbors by similarity in reverse order
            neighbors = sorted(neighbors.items(), key=operator.itemgetter(1), reverse=True)
            neighbors_by_document[target] = [document for (document, similarity) in neighbors]

        return neighbors_by_document

    def __idf__(self, term):
        """
        Inverse Document Frequency

        The inverse document frequency is a measure of how much information a term provides in relationship to a set of
        documents. The value is logarithmically scaled to give exponentially less weight to a term that is exponentially
        more informative.
        :param term: the term to calculate the inverse document frequency of.
        :return: the inverse document frequency of the term
        """

        # First check to see if we have already computed the IDF for this term
        if term in self.__idf_by_term:
            return self.__idf_by_term[term]

        # Count the frequency of each term
        freq_by_term = {}
        for document in self.documents:
            for term in set(document.text.split()):
                if term not in freq_by_term:
                    freq_by_term[term] = 1
                else:
                    freq_by_term[term] += 1

        # Calculate the Inverse Document Frequency of each term
        for term, freq in freq_by_term.iteritems():
            self.__idf_by_term[term] = 1 + math.log10(len(self.documents) / freq)

        return self.__idf_by_term[term]

    def __tf__(self, term, document):
        """
        Term Frequency

        The term frequency is a count of how often a term occurs in a document
        :param term: the term to calculate the frequency of
        :param document: the document to calculate the term frequency from
        :return: the frequency of therm within the given document
        """

        # First check to see if we have already calculated the term frequencies for this document
        if document in self.__tf_by_term_by_document:
            tf_by_term = self.__tf_by_term_by_document[document]
            return tf_by_term[term]

        # Count the frequency of each term
        freq_by_term = {}
        for term in document.text.split():
            if term not in freq_by_term:
                freq_by_term[term] = 1
            else:
                freq_by_term[term] += 1

        self.__tf_by_term_by_document[document] = freq_by_term

        return self.__tf_by_term_by_document[document][term]

    def __magnitude__(self, vector):
        """
        Determine the magnitude of a vector

        :param vector: The vector to find the magnitude of
        :return: The vector's magnitude
        """
        return math.sqrt(sum([i ** 2 for i in vector]))

    def __cosine_distance__(self, avector, bvector):
        """
        Determine the cosine distance of two vectors.

        Elements are compared by position.

        :param avector: The first vector to be compared
        :param bvector: The second vector to be compared
        :return: The cosine distance
        """
        num = sum([a * b for a, b in zip(avector, bvector)])
        den = self.__magnitude__(avector) * self.__magnitude__(bvector)
        return 1 - (num / den)

class Document(object):
    """
    Documents are used to keep relationships text and ids for lookup
    """

    def __init__(self, id, text):
        """
        A basic document object which is used to keep of text relationship. On construction the provided text is
        copied and cleaned.
        :param id: the document's id
        :param text: the provided document text.
        :return:
        """
        self.id = id
        self.text = self.__clean__(text)
        self.__original_text = text

    def __clean__(self, text):
        """
        Clean up a document through stemming and stop word removal.

        Stemming is the act of removing suffixes from a word to limit variation between verb tenses.

        Stop word remove is the act of removing common words from the document that likely play no meaning in the
        significance of the document.
        :param document: The document to be cleaned
        :return: The given document after stemming and stop word removal
        """
        stopwords = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]
        sb = []
        text = text.lower()
        re.sub(r'[\W_]+', '', text)
        for term in text.split():
            term = stem(term)
            if term not in stopwords:
                sb.append(term)

        return ' '.join(sb)
