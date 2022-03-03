import os
import argparse
from pathlib import Path
import re

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

def transform_training_data(title, comment):
    text = title + ' ' + comment
    text = re.sub('[^\w\s]', ' ', text.lower())
    if len(text)>0:
        text = ' '.join([stemmer.stem(x) for x in text.split()])
    return text

# Directory for review data
directory = r'/workspace/datasets/product_data/reviews/'
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing reviews")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--bucket_rating", action="store_true", help="whether to bucketize ratings")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input

def rating_category(r):
    # print(f'rating: {r}')
    if float(r) < 3.0:
        return 'negative'
    elif float(r) > 3.0:
        return 'positive'
    return 'neutral'

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            with open(os.path.join(directory, filename)) as xml_file:
                for line in xml_file:
                    if '<rating>'in line:
                        rating = line[12:15]
                    elif '<title>' in line:
                        title = line[11:len(line) - 9]
                    elif '<comment>' in line:
                        comment = line[13:len(line) - 11]
                    elif '</review>'in line:
                        if args.bucket_rating:
                            rating = rating_category(rating)
                        output.write("__label__%s %s\n" % (rating, transform_training_data(title, comment)))
