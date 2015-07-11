__author__ = 'tdstein'

import math
import re
import collections
from stemming.porter2 import stem


class TFIDF(object):

    documents = []

    def __init__(self, documents):
        self.documents += documents

    def add_document(self, document):
        """
        Add a single document to the training set
        """
        self.documents.append(document)

    def add_documents(self, documents):
        """
        Add a set of documents to the training set
        """
        self.documents += documents

    def evaluate(self):

        # Determine the inverse document frequency of all terms in the vocabulary
        idf = self.__idf__()

        # Determine the tfidf score for each term in each document
        tfidf_document_map = {}
        for document in self.documents:
            tfidf_map = {}
            tf = self.__tf__(document)
            for term in document.text.split():
                tfidf_map[term] = tf[term] * idf[term]
            tfidf_document_map[document] = collections.OrderedDict(sorted(tfidf_map.items()))

        most_similar_map = {}
        for k_target, v_target in tfidf_document_map.iteritems():
            distance_map = {}
            for k_neigh, v_neigh in tfidf_document_map.iteritems():
                if k_target.id is not k_neigh.id:
                    keys = set(v_target.keys()).intersection(set(v_neigh.keys()))

                    target_vector = [v for k, v in v_target.iteritems() if k in keys]
                    neigh_vector = [v for k, v in v_neigh.iteritems() if k in keys]

                    if target_vector and neigh_vector:
                        distance = self.__cosine_distance__(target_vector, neigh_vector)
                        if distance > 0:
                            distance_map[k_neigh] = distance

            distance_map = collections.OrderedDict(sorted(distance_map.items()))
            most_similar_map[k_target] = list(reversed([d for d in collections.OrderedDict(sorted(distance_map.items())).keys()]))

        return most_similar_map

    def __idf__(self):
        """
        Inverse Document Frequency

        The inverse document frequency is a measure of how much information a term provides in relationship to a set of
        documents. The value is logarithmically scaled to give exponentially less weight to a term that is exponentially
        more informative.
        :return:
        """
        term_counts = {}
        for document in self.documents:
            for term in set(document.text.split()):
                if term not in term_counts:
                    term_counts[term] = 1
                else:
                    term_counts[term] += 1

        num_of_documents = len(self.documents)
        for k, v in term_counts.iteritems():
            term_counts[k] = 1 + math.log10((num_of_documents / v))

        return term_counts

    def __tf__(self, document):
        terms = document.text.split()
        term_counts = {}
        for term  in terms:
            if term not in term_counts:
                term_counts[term] = 1
            else:
                term_counts[term] += 1

        return term_counts

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

# python = """
# Python is a widely used general-purpose, high-level programming language.[19][20][21] Its design philosophy emphasizes code readability, and its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java.[22][23] The language provides constructs intended to enable clear programs on both a small and large scale.[24]
# Python supports multiple programming paradigms, including object-oriented, imperative and functional programming or procedural styles. It features a dynamic type system and automatic memory management and has a large and comprehensive standard library.[25]
# Python interpreters are available for installation on many operating systems, allowing Python code execution on a wide variety of systems. Using third-party tools, such as Py2exe or Pyinstaller,[26] Python code can be packaged into stand-alone executable programs for some of the most popular operating systems, allowing for the distribution of Python-based software for use on those environments without requiring the installation of a Python interpreter.
# CPython, the reference implementation of Python, is free and open-source software and has a community-based development model, as do nearly all of its alternative implementations. CPython is managed by the non-profit Python Software Foundation.
# """
#
# ruby = """
# Ruby is a dynamic, reflective, object-oriented, general-purpose programming language. It was designed and developed in the mid-1990s by Yukihiro "Matz" Matsumoto in Japan.
# According to its authors, Ruby was influenced by Perl, Smalltalk, Eiffel, Ada, and Lisp.[13] It supports multiple programming paradigms, including functional, object-oriented, and imperative. It also has a dynamic type system and automatic memory management.
# """
#
# scala = """
# Scala is a programming language for general software applications. Scala has full support for functional programming and a very strong static type system. This allows programs written in Scala to be very concise and thus smaller in size than other general-purpose programming languages. Many of Scala's design decisions were inspired by criticism of the shortcomings of Java.[5]
# Scala source code is intended to be compiled to Java bytecode, so that the resulting executable code runs on a Java virtual machine. Java libraries may be used directly in Scala code and vice versa (language interoperability).[7] Like Java, Scala is object-oriented, and uses a curly-brace syntax reminiscent of the C programming language. Unlike Java, Scala has many features of functional programming languages like Scheme, Standard ML and Haskell, including currying, type inference, immutability, lazy evaluation, and pattern matching. It also has an advanced type system supporting algebraic data types, covariance and contravariance, higher-order types (but not higher-rank types), and anonymous types. Other features of Scala not present in Java include operator overloading, optional parameters, named parameters, raw strings, and no checked exceptions.
# The name Scala is a portmanteau of "scalable" and "language", signifying that it is designed to grow with the demands of its users.[8]
# """
#
# java = """
# Java is a general-purpose computer programming language that is concurrent, class-based, object-oriented,[12] and specifically designed to have as few implementation dependencies as possible. It is intended to let application developers "write once, run anywhere" (WORA),[13] meaning that compiled Java code can run on all platforms that support Java without the need for recompilation.[14] Java applications are typically compiled to bytecode that can run on any Java virtual machine (JVM) regardless of computer architecture. As of 2015, Java is one of the most popular programming languages in use,[15][16][17][18] particularly for client-server web applications, with a reported 9 million developers.[citation needed] Java was originally developed by James Gosling at Sun Microsystems (which has since been acquired by Oracle Corporation) and released in 1995 as a core component of Sun Microsystems' Java platform. The language derives much of its syntax from C and C++, but it has fewer low-level facilities than either of them.
# The original and reference implementation Java compilers, virtual machines, and class libraries were originally released by Sun under proprietary licences. As of May 2007, in compliance with the specifications of the Java Community Process, Sun relicensed most of its Java technologies under the GNU General Public License. Others have also developed alternative implementations of these Sun technologies, such as the GNU Compiler for Java (bytecode compiler), GNU Classpath (standard libraries), and IcedTea-Web (browser plugin for applets).
# """
#
# documents = []
# documents.append(Document("python", python))
# documents.append(Document("ruby", ruby))
# documents.append(Document("scala", scala))
# documents.append(Document("java", java))
#
# test = TFIDF(documents)
#
# print test.evaluate(len(documents))
