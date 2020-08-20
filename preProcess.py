import json
import re
import sys

import nltk
import numpy
import pandas
from nltk.corpus import stopwords
from langdetect import detect

business_input = "/Users/anshulgupta/Desktop/DataMining/business.json"
# business_input = str(sys.argv[1])

review_input = "/Users/anshulgupta/Desktop/DataMining/review.json"
# review_input = str(sys.argv[2])

output = "newOutput2.csv"
# output file


def preProcessing(text, variation):
    text = text.lower().split()
    new_text = []
    for word in text:
        if word in variation:
            new_text.append(variation[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


business_id = []
nltk.download('stopwords')
business_input_file = open(business_input, 'r')
for line in business_input_file:
    business_data = json.loads(line)
    if 'categories' in business_data and business_data['categories'] is not None:
        # taking only restaurant and food data
        if 'Restaurants' in business_data['categories'] or 'Food' in business_data['categories']:
            business_id.append(business_data['business_id'])

print("restaurants/food: " + str(len(business_id)))

# downloaded from internet: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
mapping = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

reviews_count = 0
review_input_file = open(review_input, 'r')
updated_text = []
input_stars = []
for line in review_input_file:
    review_data = json.loads(line)
    bus_id = -1
    review_text = ""
    star_rating = -1.0
    if 'business_id' in review_data:
        bus_id = review_data['business_id']
    if 'text' in review_data:
        review_text = review_data['text']
    if 'stars' in review_data:
        star_rating = review_data['stars']

    if not (not (bus_id == -1) and not (bus_id not in business_id) and not (len(review_text) == 0) and not (
            star_rating == -1.0) and not (review_text == ";))") and not (review_text == ":)")):
        continue
    lan = detect(review_text)
    if lan == 'en':
        final_text = preProcessing(review_text, mapping)
        updated_text.append(final_text)
        input_stars.append(star_rating)
        reviews_count += 1

    if reviews_count == 100000:
        break

    if reviews_count % 10000 == 0:
        print(reviews_count)

final_array = numpy.hstack((numpy.asarray([updated_text]).T, numpy.asarray([input_stars]).T))
col = ['text', 'stars']
df_final_array = pandas.DataFrame(final_array, columns=col)
df_final_array[['stars']] = df_final_array[['stars']].apply(pandas.to_numeric)
df_stars = df_final_array[numpy.isfinite(df_final_array['stars'])]
df_final_stars = df_stars.dropna()
df_final_stars = df_final_stars.reset_index(drop=True)
df_final_stars['len'] = df_final_stars.text.str.len()
df_final_stars = df_final_stars[df_final_stars['len'].between(10, 4000)]
col2 = ['stars']
lowest_count = df_final_stars.groupby(col2).apply(lambda x: x.shape[0]).min()
df_final = df_final_stars.groupby(col2).apply(
    lambda x: x.sample(lowest_count)).drop(col2, axis=1).reset_index().set_index('level_1').sort_index(inplace=True)

df_final.to_csv(output, encoding='utf-8')
print(len(df_final['stars']))
