#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

# import nltk
import re
from deep_translator import GoogleTranslator
from collections import Counter
from konlpy.corpus import kolaw
from konlpy.tag import Kkma
from konlpy.utils import concordance, pprint
from matplotlib import pyplot

# nltk.download('punkt')

def draw_zipf(count_list, filename, color='blue', marker='o'):
    sorted_list = sorted(count_list, reverse=True)
    pyplot.plot(sorted_list, color=color, marker=marker)
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.savefig(filename)


doc = kolaw.open('constitution.txt').read()
output = re.sub(r"\u2460|\u2461|\u2462|\u2463|\u2464|\u2465|\u2466|\u2467", "", doc)
# with open("original_txt.txt", 'w', encoding="utf-8") as original_doc:
#     original_doc.write(doc)

with open("translation.txt", 'w') as translated_doc:
    tokens = output.split("\n")
    for token in tokens:
        print(GoogleTranslator(source='ko', target='en').translate(token), file=translated_doc)


pos = Kkma().pos(doc)
cnt = Counter(pos)

print('nchars  :', len(doc))
print('ntokens :', len(doc.split()))
print('nmorphs :', len(set(pos)))
print('\nTop 20 frequent morphemes:'); pprint(cnt.most_common(20))
print('\nLocations of "대한민국" in the document:')
concordance(u'대한민국', doc, show=True)

draw_zipf(cnt.values(), 'zipf.png')