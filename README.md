# Generate a wordcloud normalizing, stemming, n-gram'ing words from an input file

This repo contains example code of how to create a WordCloud by applying some common NLP techniques to the text first.  

## Overview

When trying to understand how a machine learning algorithm might process a corpus of text, such a linkedin profile, or a resume, it is helpful to understand how the text data will be processed.

Some typical pre-processing steps include:

- Removing stop words.  Stop words are those words that have been determined to add to the 'noise' which obfuscates the important words, aka signal. 

- Lemmatizing words.  Lemmatization is the process of changing a word to its dictionary form.  For example, cats becomes cat and geese becomes goose.

At the end of these steps, only the non-stopwords are returned, and those words are transformed into the dictionary form.  In this way there is a consistent base if you are comparing documents.

## Visualizing the Text Normalizing

The goal is to create a WordCloud of only the important words to create a visual representation of what a Machine Learning algorithm might see.

By 'important', at this point we are just creating a 'count' of the words after all of the stop words have been removed, stemming the remaining words and adding ngrams to the list.

ngrams, is a term meant to indicate the number of words we consider together.  For example a 2-gram means we look at pairs of words in addition to each work individually.

Why is this important?  Imagine your text had the word 'machine' and 'learning'.  Individually these words could convey that you are learning about how machines work. But taken together, 'machine learning' - indicates the study of how machine learn from data.

## Code Details

### Setup

This code was tested with Python 3.6, but I see no reason with it would not work with 3.6+

- Clone the repo

- Create a python virtual env

- pip install -r requirements.txt

- execute:

```bash
python wordcloud_util.py
```

### Input Text

By default the script is designed to read text from `input_text.txt`.  To see what your data might look like update the text in that file.


### Image Output

The script is setup to create a png file called:  `wordcloud_image.png`

### TextNormalizer.py

This class leverages `nltk` to process the raw text into a collection of strings.  The strings are those that are not part of hte stop words and have been lemmatized.

In addition, it creates 2-gram words by taking each 2-gram token and concatinating with an underscore.  For example a 2-gram of ('machine', 'leaning') becomes a 1-gram word of 'machine_learning'


