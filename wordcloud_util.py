from wordcloud import WordCloud
from TextNormalizer import TextNormalizer
from wordcloud import STOPWORDS
from os import path
from pathlib import Path

text_normalizer = TextNormalizer()
twitter_stop_words = ['rt', 'ty', 'de', 'co', 'your', 'use', 'help', 'RT', 'http', 'https']
jobs_stop_words = ['xa0', 'take', 'brand', 'role', 'mission', 'create', 'report', 'seek', 'fast', 'tool', 'sale',
                   'marketing', 'business', 'exist', 'grow', 'best', 'ensure', 'user', 'within', 'time', 'provide',
                   'network', 'review']
text_normalizer.add_stopwords(twitter_stop_words)
text_normalizer.add_stopwords(jobs_stop_words)
text_normalizer.add_stopwords(STOPWORDS)
from collections import Counter


def create_word_cloud_from_document(doc, imagedir, filename):
    text_normalizer = TextNormalizer()
    twitter_stop_words = ['rt', 'ty', 'de', 'co', 'your', 'use', 'help', 'RT', 'http', 'https']
    jobs_stop_words = ['xa0', 'take', 'brand', 'role', 'mission', 'create', 'report', 'seek', 'fast', 'tool', 'sale',
                       'marketing', 'business', 'exist', 'grow', 'best', 'ensure', 'user', 'within', 'time', 'provide',
                       'network', 'review']
    text_normalizer.add_stopwords(twitter_stop_words)
    text_normalizer.add_stopwords(jobs_stop_words)
    text_normalizer.add_stopwords(STOPWORDS)

    normalized_words = text_normalizer.normalize(doc, ngram_size=2)

    c = Counter(normalized_words)

    wordcloud = WordCloud(width=1000, height=860, max_words=100).generate_from_frequencies(dict(c))
    wordcloud.to_file(path.join(imagedir, filename))


if __name__ == '__main__':
    input_text = Path("./input_text.txt").read_text()

    create_word_cloud_from_document(input_text, "./", "wordcloud_image.png")
