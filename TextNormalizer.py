import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk import wordpunct_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk import ngrams


class TextNormalizer():

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(["said", "&amp;", "ok?", "", 'amp'])
        self.stopwords.update(["&lt;", "&gt;", "&quot;", "&amp;", "w/o", "w/"])
        self.stopwords.update(
            ['him?', 'now,', '-', 'see', 'go', 'great".', "he's", 'got', 'ok?', 'gets', 'get', 'for', 'on', 'one', 'he',
             'we', 'want', 'it.', 'O.', 'and', 'the', "will", "say", "said", "new", "day", "he's", 'But', 'A', 'They',
             'The', 'I', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
             "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
             'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
             'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
             'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
             'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
             'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
             'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
             'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
             'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
             "won't", 'wouldn', "wouldn't"])

        self.tokenizer = wordpunct_tokenize
        self.lemmatizer = WordNetLemmatizer()

    def add_stopwords(self, new_stop_words):
        self.stopwords.update(new_stop_words)

    def _is_stopword(self, token):
        return token.lower() in self.stopwords or len(token) == 1 or token.isdigit()

    def _remove_html_lower(self, text):
        # remove url
        text = re.sub(r"http\S+", "", text)

        text = re.sub('<[^>]*', '', text)

        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

        text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

        return text

    def _lemmatize(self, token, pos_token_tag):
        """
        1.	CC	Coordinating conjunction
    2.	CD	Cardinal number
    3.	DT	Determiner
    4.	EX	Existential there
    5.	FW	Foreign word
    6.	IN	Preposition or subordinating conjunction
    7.	JJ	Adjective
    8.	JJR	Adjective, comparative
    9.	JJS	Adjective, superlative
    10.	LS	List item marker
    11.	MD	Modal
    12.	NN	Noun, singular or mass
    13.	NNS	Noun, plural
    14.	NNP	Proper noun, singular
    15.	NNPS	Proper noun, plural
    16.	PDT	Predeterminer
    17.	POS	Possessive ending
    18.	PRP	Personal pronoun
    19.	PRP$	Possessive pronoun
    20.	RB	Adverb
    21.	RBR	Adverb, comparative
    22.	RBS	Adverb, superlative
    23.	RP	Particle
    24.	SYM	Symbol
    25.	TO	to
    26.	UH	Interjection
    27.	VB	Verb, base form
    28.	VBD	Verb, past tense
    29.	VBG	Verb, gerund or present participle
    30.	VBN	Verb, past participle
    31.	VBP	Verb, non-3rd person singular present
    32.	VBZ	Verb, 3rd person singular present
    33.	WDT	Wh-determiner
    34.	WP	Wh-pronoun
    35.	WP$	Possessive wh-pronoun
    36.	WRB	Wh-adverb
        :param token:
        :param pos_tag:
        :return:
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_token_tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def normalize(self, document, ngram_size=1):
        doc2 = self._remove_html_lower(str(document))
        tokens = self.tokenizer(doc2)

        x = [t for t in tokens if not self._is_stopword(t)]

        if ngram_size > 1:
            ngram_tokens = ngrams(x.copy(), ngram_size)
            for ngram_token in ngram_tokens:
                # print(ngram_token)
                token = '_'.join(ngram_token)
                x.append(token)

        normalized_tokens = [self._lemmatize(pt[0], pt[1]) for pt in pos_tag(x)]

        x_normal = [t for t in normalized_tokens if not self._is_stopword(t)]

        return x_normal