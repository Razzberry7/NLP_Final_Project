#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

# import nltk
from nltk import collocations, RegexpParser, pos_tag, word_tokenize, FreqDist
import re
from deep_translator import GoogleTranslator
from collections import Counter
from konlpy.corpus import kolaw
from konlpy.tag import Kkma, Okt
from konlpy.utils import concordance, pprint
from matplotlib import pyplot
import matplotlib.font_manager as fm
import os
from alive_progress import alive_bar; import time
import random
import string
from Korpora import Korpora
import numpy
import json
import sys
from enum import Enum
import csv

# Method to print concordance to file (adapted from KoNLPy's util.py) 
def concordance_to_file(phrase, text, show, output_file):
    terms = text.split()
    indexes = [i for i, term in enumerate(terms) if phrase in term]
    if show:
        for i in indexes:
            print('%d\t%s' % (i, ' '.join(terms[max(0, i - 3):i + 3])), file=output_file)
    return indexes

# Method to create a Zipf plot (copied from a KoNLPy example)
def draw_zipf(count_list, filename, color='blue', marker='o'):
    sorted_list = sorted(count_list, reverse=True)
    pyplot.plot(sorted_list, color=color, marker=marker)
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.savefig(filename)

# Method to create folders for organization purposes
def create_folders(verbose=False):
    # Create Needed folders to hold corpora
    if not os.path.exists("./Korean/With_Stopwords/"):
        os.makedirs("./Korean/With_Stopwords/")
    elif verbose:
        print("The directory at ./Korean/With_Stopwords/ already exists!")

    if not os.path.exists("./Korean/Without_Stopwords/"):
        os.makedirs("./Korean/Without_Stopwords/")
    elif verbose:
        print("The directory at ./Korean/Without_Stopwords/ already exists!")

    if not os.path.exists("./English/With_Korean_Stopwords/"):
        os.makedirs("./English/With_Korean_Stopwords/")
    elif verbose:
        print("The directory at ./English/With_Korean_Stopwords/ already exists!")

    if not os.path.exists("./English/Without_Korean_Stopwords/"):
        os.makedirs("./English/Without_Korean_Stopwords/")
    elif verbose:
        print("The directory at ./English/Without_Korean_Stopwords/ already exists!")

    if not os.path.exists("./Charts/"):
        os.makedirs("./Charts/")
    elif verbose:
        print("The directory at ./Charts/ already exists!")

    if not os.path.exists("./Tables/"):
        os.makedirs("./Tables/")
    elif verbose:
        print("The directory at ./Tables/ already exists!")

### MAIN METHOD ###
# Method to clean text, save text to files, translate the text, and perform analysis
def start_pipeline(corpus_title, corpus_text, gen_seed):
    # Take out unicode symbols
    doc_without_unicode_chars = remove_unicodes_ko(corpus_text=corpus_text)

    # Count important elments of current version of Korean corpus (No stopwords removed)
    count_korean_elements(corpus_title=f"{corpus_title}_with_stopwords", corpus_text=doc_without_unicode_chars, path="./Korean/With_Stopwords", recreate=True, gen_seed=gen_seed)

    # Save the original Korean corpus text 
    save_as_txt(corpus_title=corpus_title, corpus_text=doc_without_unicode_chars, save_path="./Korean/With_Stopwords")

    # Translate current version of corpus (No stopwords removed)
    translated_corpus_with_stopwords = translate_corpus_ko_to_en(corpus_title=corpus_title, corpus_text=doc_without_unicode_chars, path="./English/With_Korean_Stopwords")

    # Count important elments of current version of English corpus (No stopwords removed)
    count_english_elements(corpus_title=f"{corpus_title}_with_stopwords", corpus_text=translated_corpus_with_stopwords, path="./English/With_Korean_Stopwords", recreate=True, gen_seed=gen_seed)

    # Take out Korean stopwords
    doc_without_stopwords = remove_stopwords_ko(corpus_title=corpus_title, corpus_text=doc_without_unicode_chars)

    # Save this version of the Korean corpus (with no stopwords)
    save_as_txt(corpus_title=corpus_title, corpus_text=doc_without_stopwords, save_path="./Korean/Without_Stopwords")

    # Count important elments of current version of Korean corpus (Korean stopwords removed)
    count_korean_elements(corpus_title=f"{corpus_title}_without_stopwords", corpus_text=doc_without_stopwords, path="./Korean/Without_Stopwords", recreate=True, gen_seed=gen_seed)

    # Translate current version of corpus (Korean stopwords removed)
    translated_corpus_without_stopwords = translate_corpus_ko_to_en(corpus_title=corpus_title, corpus_text=doc_without_stopwords, path="./English/Without_Korean_Stopwords")

    # Count important elments of current version of English corpus (Korean stopwords removed)
    count_english_elements(corpus_title=f"{corpus_title}_without_stopwords", corpus_text=translated_corpus_without_stopwords, path="./English/Without_Korean_Stopwords", recreate=True, gen_seed=gen_seed)

# Method to remove unicodes that break the deep_translator
def remove_unicodes_ko(corpus_text):
    print("Taking out some unicode characters...")
    
    # Each element here broke the translator, but Korean characters are unicode, so we can't remove all of them
    return re.sub(r"\u2460|\u2461|\u2462|\u2463|\u2464|\u2465|\u2466|\u2467|;+", "", corpus_text)  

# Method to remove Korean stopwords (See README for source of stopwords list)
def remove_stopwords_ko(corpus_title, corpus_text):
    print(f"Starting to take stopwords out of original_{corpus_title}.txt !")
    with open('korean_stopwords.txt', encoding="utf8") as ksw:
        stopwords = ksw.read().splitlines()
        for stopword in stopwords:
            if stopword in corpus_text.split():
                # print(f"Removing the word: {stopword}")
                corpus_text = corpus_text.replace(stopword, '')
    return corpus_text

# Method to save text as a file
def save_as_txt(corpus_title, corpus_text, save_path):
    print("Saving this version of the Korean corpus...")
    file_to_create = f"{save_path}/original_{corpus_title}.txt"
    if not os.path.exists(file_to_create):
            print(f"Writing original_{corpus_title}.txt to {save_path} !")
            with open(f"{save_path}/original_{corpus_title}.txt", 'w', encoding="utf-8") as original_doc:
                original_doc.write(corpus_text)
    else:
        print(f"\t{file_to_create} already exists! Skipping this file.")

# Method to translate a Korean corpus to English using deep_translator's GoogleTranslator
def translate_corpus_ko_to_en(corpus_title, corpus_text, path):
    file_to_create = f"{path}/translation_{corpus_title}.txt"
    if not os.path.exists(file_to_create):
        print(f"Translating and writing translation_{corpus_title}.txt to {path} !")
        with open(f"{path}/translation_{corpus_title}.txt", 'w', encoding="utf-8") as translated_doc:
            # Split up the corpus into lines to not overwhelm the translator
            tokens = corpus_text.split("\n")
            # Create a progress bar to see current translation progress
            with alive_bar(len(tokens)) as bar:
                for token in tokens:
                    translation = GoogleTranslator(source='ko', target='en').translate(token)
                    bar()
            translated_doc.write(translation)
        return translation
    else:
        print(f"\t{file_to_create} already exists! Skipping this file.")
        # If the file already exists, just read the existing translation and return that
        with open(f"{path}/translation_{corpus_title}.txt", 'r', encoding="utf-8") as translated_doc:
            translation = translated_doc.read()
        return translation

# Method to count important elements of a Korean corpus
def count_korean_elements(corpus_title, corpus_text, path, recreate=False, gen_seed="No"):
    file_to_create = f"{path}/analysis_{corpus_title}.txt"
    if not os.path.exists(file_to_create) or recreate is True:
        print("Begin counting important elements in the Korean corpus...")

        if corpus_title.endswith("with_stopwords"):
            current_version = "with_stopwords"
        elif corpus_title.endswith("without_stopwords"):
            current_version = "without_stopwords"
        else:
            print("This corpus is neither version (with/without stopwords)! Something is wrong...")
            sys.exit()

        # Use Okt to grab all the phrases in the text and count them up
        print("Grabbing and counting phrases...")
        phrases = Okt().phrases(corpus_text)
        phrase_count = Counter(phrases)

        # Use Okt to grab all the nouns in the text and count them up
        print("Grabbing and counting nouns...")
        nouns = Okt().nouns(corpus_text)
        noun_count = Counter(nouns)

        # Use Okt to grab all the verbs in the text and count them up
        print("Grabbing and counting verbs...")
        verbs = Okt().verbs(corpus_text)
        verb_count = Counter(verbs)

        # Use Okt to grab all the adverbs in the text and count them up
        print("Grabbing and counting adverbs...")
        adverbs = Okt().adverbs(corpus_text)
        adverb_count = Counter(adverbs)

        # Use Okt to grab all the adjectives in the text and count them up
        print("Grabbing and counting adjectives...")
        adjectives = Okt().adjectives(corpus_text)
        adjective_count = Counter(adjectives)

        # Use Okt to grab all the Josa (Korean particles) in the text and count them up
        print("Grabbing and counting josa (Korean particles)...")
        josas = Okt().josa(corpus_text)
        josa_count = Counter(josas)

        # Use Okt to grab all the parts of speech in the text and count them up
        print("Grabbing and counting all POS...")
        pos = Okt().pos(corpus_text)
        pos_count = Counter(pos)

        # Create a bigram collocation object
        print("Initializing collocations measure and finding collocations...")
        measures = collocations.BigramAssocMeasures()
        finder = collocations.BigramCollocationFinder.from_words(pos)


        # Save these results to a file
        print("Saving data to file...")
        with open(f"{path}/analysis_{corpus_title}.txt", 'w', encoding="utf-8") as analysis_txt:
            # Counts of various elements in the text
            num_of_chars = len(corpus_text)
            print(f'Number of characters in {corpus_title}.txt:', num_of_chars, file=analysis_txt)
            num_of_words = len(corpus_text.split())
            print(f'Number of words in {corpus_title}.txt:', num_of_words, file=analysis_txt)
            
            # Note: phrases is loosely defined by the Okt class
            print(f'\nTop 10 frequent phrases in {corpus_title}.txt:', file=analysis_txt) 
            pprint(phrase_count.most_common(10), stream=analysis_txt) 

            # Note: contains punctuation, including "\n" but this is needed to keep lines separate for translation
            print(f'\nTop 10 frequent morphemes in {corpus_title}.txt:', file=analysis_txt); 
            pprint(pos_count.most_common(10), stream=analysis_txt)
            print(f'Number of morphemes in {corpus_title}.txt:', len(set(pos)), file=analysis_txt)

            print(f'\nTop 10 frequent nouns in {corpus_title}.txt:', file=analysis_txt) 
            pprint(noun_count.most_common(10), stream=analysis_txt)
            num_of_nouns = len(noun_count)
            print(f'Number of nouns in {corpus_title}.txt:', num_of_nouns, file=analysis_txt)

            print(f'\nTop 10 frequent verbs in {corpus_title}.txt:', file=analysis_txt) 
            pprint(verb_count.most_common(10), stream=analysis_txt)
            num_of_verbs = len(verb_count)
            print(f'Number of verbs in {corpus_title}.txt:', num_of_verbs, file=analysis_txt)

            print(f'\nTop 10 frequent adverbs in {corpus_title}.txt:', file=analysis_txt) 
            pprint(adverb_count.most_common(10), stream=analysis_txt)
            num_of_adverbs = len(adverb_count)
            print(f'Number of adverbs in {corpus_title}.txt:', num_of_adverbs, file=analysis_txt)

            print(f'\nTop 10 frequent adjectives in {corpus_title}.txt:', file=analysis_txt) 
            pprint(adjective_count.most_common(10), stream=analysis_txt)
            num_of_adjectives = len(adjective_count)
            print(f'Number of adjectives in {corpus_title}.txt:', num_of_adjectives, file=analysis_txt)

            print(f'\nTop 10 frequent josa (Korean particles) in {corpus_title}.txt:', file=analysis_txt) 
            pprint(josa_count.most_common(10), stream=analysis_txt)
            num_of_josa = len(josa_count)
            print(f'Number of josa (Korean Particles) in {corpus_title}.txt:', num_of_josa, file=analysis_txt)

            # These are collocations amount the tagged words, not general collocations
            print('\nCollocations among tagged words:', file=analysis_txt)
            pprint(finder.nbest(measures.pmi, 10), stream=analysis_txt) # top 10 n-grams with highest PMI

            sample_size=5
            print(f"Chunking {sample_size} randomly sampled sentences and saving to file...")
            sampled_sentences = []
            for sentence in randomly_sample(list=corpus_text.split('\n'), sample_size=sample_size, gen_seed=gen_seed):
                print(f"\nRandomly sampling this sentence for chunking: {sentence}", file=analysis_txt)

                sampled_sentences.append(sentence)

                words = Okt().pos(sentence)

                # Define a chunk grammar, or chunking rules, then chunk
                grammar = """
                NP: {<N.*>*<Suffix>?}   # Noun phrase
                VP: {<V.*>*}            # Verb phrase
                AP: {<A.*>*}            # Adjective phrase
                """
                
                # Parse the sampled sentences following the grammar set above
                parser = RegexpParser(grammar)
                chunks = parser.parse(words)
                print(f"Whole chunk tree for sentence: {sentence}", file=analysis_txt)
                print(chunks, file=analysis_txt)

                # Note: This is the only way to display the tree graphically, as Korean characters don't
                # seem to save to postscript files when using the appropriate NLTK libraries to save the tree.
                # This is commented out for general use cases, but left in for when I want to grab some screenshots
                # of trees as examples for the paper.
                ###chunks.draw() 

            # Note: could be interesting to search for specific words, but not important for project goals
            # print(f'\nLocations of "대한민국" in {corpus_title}.txt:', file=analysis_txt)
            # concordance_to_file(phrase=u'대한민국', text=corpus_text, show=True, output_file=analysis_txt)

            # # Draw a Zipf's Law plot
            # print("Drawing a Zipf's Law plot...")
            # draw_zipf(count_list=pos_count.values(), filename=f'{path}/{corpus_title}_zipf.png')

            try:
                # Load existing data
                with open("./corpus_data.json", 'r') as existing_json:
                    existing_data = json.load(existing_json)
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is empty, start a new dictionary
                existing_data = {"overall_analysis": 
                                 {"english" : {"with_stopwords" : {}, 
                                               "without_stopwords" : {}}, 
                                  "korean": {"with_stopwords" : {}, 
                                             "without_stopwords" : {}}}}
                
            new_data_to_save_to_json = {
                corpus_title : {
                        "char_count" : num_of_chars,
                        "word_count" : num_of_words,
                        "noun_count" : num_of_nouns,
                        "verb_count" : num_of_verbs,
                        "adverb_count" : num_of_adverbs,
                        "adjective_count" : num_of_adjectives,
                        "particle_count" : num_of_josa,
                        "sampled_sentences" : sampled_sentences
                }
            }

            existing_data["overall_analysis"]["korean"][current_version].update(new_data_to_save_to_json)

            with open (f"./corpus_data.json", 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)

    else:
        print(f"\t{file_to_create} already exists! Recreate flag was not specified, this file will be skipped.")

# Method to count important elements of an English corpus
def count_english_elements(corpus_title, corpus_text, path, recreate=False, gen_seed="No"):  
    file_to_create = f"{path}/analysis_{corpus_title}.txt"
    if not os.path.exists(file_to_create) or recreate is True:
        print("Begin counting important elements in the English corpus...")

        if corpus_title.endswith("with_stopwords"):
            current_version = "with_stopwords"
        elif corpus_title.endswith("without_stopwords"):
            current_version = "without_stopwords"
        else:
            print("This corpus is neither version (with/without stopwords)! Something is wrong...")
            sys.exit()

        # Tokenize corpus for tagging parts of speech
        tokenized_corpus = word_tokenize(corpus_text)
        tagged_corpus = pos_tag(tokens=tokenized_corpus)

        # Compile lists of different parts of speech
        print("Counting nouns, verbs, adverbs, adjectives, and particles...")
        nouns, verbs, adverbs, adjectives, particles = [], [], [], [], [] 
        for word, tag in tagged_corpus:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                nouns.append(word)
            elif tag in ['VB', 'VBD', 'VBG', 'VBP', 'VBZ']:
                verbs.append(word)
            elif tag in ['RB', 'RBR', 'RBS']:
                adverbs.append(word)
            elif tag in ['JJ', 'JJR', 'JJS']:
                adjectives.append(word)
            elif tag in ['RP']:
                particles.append(word)

        # Create a bigram collocation object
        print("Initializing collocations measure and finding collocations...")
        measures = collocations.BigramAssocMeasures()
        finder = collocations.BigramCollocationFinder.from_words(tagged_corpus)

        # Save these results to a file
        print("Saving data to file...")
        with open(f"{path}/analysis_{corpus_title}.txt", 'w', encoding="utf-8") as analysis_txt:
            # Counts of various elements in the text
            num_of_chars = len(corpus_text)
            print(f'Number of characters in {corpus_title}.txt:', num_of_chars, file=analysis_txt)
            num_of_words = len(corpus_text.split())
            print(f'Number of words in {corpus_title}.txt:', num_of_words, file=analysis_txt)
            
            print(f"\nTop 10 frequent morphemes in {corpus_title}.txt:", file=analysis_txt)
            for common_morpheme in FreqDist(tagged_corpus).most_common(10):
                print(common_morpheme, file=analysis_txt)
            num_of_morphemes = len(tagged_corpus)
            print(f"\nNumber of particles in {corpus_title}.txt:", num_of_morphemes, file=analysis_txt)

            print(f"\nTop 10 frequent nouns in {corpus_title}.txt:", file=analysis_txt)
            for common_noun in FreqDist(nouns).most_common(10):
                print(common_noun, file=analysis_txt)
            num_of_nouns = len(nouns)
            print(f"\nNumber of nouns in {corpus_title}.txt:", num_of_nouns, file=analysis_txt)

            print(f"\nTop 10 frequent verbs in {corpus_title}.txt:", file=analysis_txt)
            for common_verb in FreqDist(verbs).most_common(10):
                print(common_verb, file=analysis_txt)
            num_of_verbs = len(verbs)
            print(f"\nNumber of verbs in {corpus_title}.txt:", num_of_verbs, file=analysis_txt)

            print(f"\nTop 10 frequent adverbs in {corpus_title}.txt:", file=analysis_txt)
            for common_adverb in FreqDist(adverbs).most_common(10):
                print(common_adverb, file=analysis_txt)
            num_of_adverbs = len(adverbs)
            print(f"\nNumber of adverbs in {corpus_title}.txt:", num_of_adverbs, file=analysis_txt)

            print(f"\nTop 10 frequent adjectives in {corpus_title}.txt:", file=analysis_txt)
            for common_adjective in FreqDist(adjectives).most_common(10):
                print(common_adjective, file=analysis_txt)
            num_of_adjectives = len(adjectives)
            print(f"\nNumber of adjectives in {corpus_title}.txt:", num_of_adjectives, file=analysis_txt)

            print(f"\nTop 10 frequent particles in {corpus_title}.txt:", file=analysis_txt)
            for common_particle in FreqDist(particles).most_common(10):
                print(common_particle, file=analysis_txt)
            num_of_particles = len(particles)
            print(f"\nNumber of particles in {corpus_title}.txt:", num_of_particles, file=analysis_txt)

            # These are collocations only among the tagged words, not general collocations
            print('\nCollocations among tagged words:', file=analysis_txt)
            pprint(finder.nbest(measures.pmi, 10), stream=analysis_txt) # top 10 n-grams with highest PMI

            print(f"Translating and Chunking the randomly sampled Korean sentences and saving to file...")
            translated_sentences = []
            with open("./corpus_data.json", 'r') as existing_json:
                existing_data = json.load(existing_json)

                # Use the current data in the JSON file to grab the sampled Korean sentences for translating/chunking
                for korean_sentence in existing_data["overall_analysis"]["korean"][current_version][corpus_title]["sampled_sentences"]:
                    print(f"\nTranslating this randomly sampled korean sentence: {korean_sentence}", file=analysis_txt)

                    translated_sentence = GoogleTranslator(source='ko', target='en').translate(korean_sentence)

                    print(f"\nChunking this translated sentence: {translated_sentence}", file=analysis_txt)

                    translated_sentences.append(translated_sentence)

                    words = pos_tag(word_tokenize(translated_sentence))

                    # Define a chunk grammar, or chunking rules, then chunk
                    grammar = """
                    NP: {<N.*>*<Suffix>?}   # Noun phrase
                    VP: {<V.*>*}            # Verb phrase
                    AP: {<A.*>*}            # Adjective phrase
                    """
    
                    # Parse the sampled sentences following the grammar set above
                    parser = RegexpParser(grammar)
                    chunks = parser.parse(words)
                    print(f"Whole chunk tree for sentence: {translated_sentence}", file=analysis_txt)
                    print(chunks, file=analysis_txt)

                    # This is commented out for general use cases, but left in for when I want to grab some screenshots
                    # of trees as examples for the paper.
                    ###chunks.draw()
                
            try:
                # Load existing data
                with open("./corpus_data.json", 'r') as existing_json:
                    existing_data = json.load(existing_json)
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is empty, start a new dictionary
                existing_data = {"overall_analysis": 
                                 {"english" : {"with_stopwords" : {}, 
                                               "without_stopwords" : {}}, 
                                  "korean": {"with_stopwords" : {}, 
                                             "without_stopwords" : {}}}}

            new_data_to_save_to_json = {
                corpus_title : {
                        "char_count" : num_of_chars,
                        "word_count" : num_of_words,
                        "noun_count" : num_of_nouns,
                        "verb_count" : num_of_verbs,
                        "adverb_count" : num_of_adverbs,
                        "adjective_count" : num_of_adjectives,
                        "particle_count" : num_of_particles,
                        "sampled_sentences" : translated_sentences
                }
            }

            existing_data["overall_analysis"]["english"][current_version].update(new_data_to_save_to_json)

            with open (f"./corpus_data.json", 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
    else:
        print(f"\t{file_to_create} already exists! Recreate flag was not specified, this file will be skipped.") 


# Method to generate a random seed of specified length; saves to seed.txt
def generate_seed(gen_seed="No", seed_length=8):
    # Generate a random seed that is 8 characters
    if gen_seed == "No" and os.path.exists("./seed.txt"):
        with open ("./seed.txt", "r") as seed_txt:
            seed = seed_txt.read()
    elif gen_seed == "Yes" or not os.path.exists("./seed.txt"):
        with open("./seed.txt", 'w', encoding="utf-8") as seed_txt:
            seed = ''.join(random.choice(string.ascii_letters + string.digits) for character in range(seed_length))
            print(f"{seed}", file=seed_txt)

    print(f"\tUsing the random seed: {seed}")
    return seed

# Method to randomly sample N sentences from a corpus
def randomly_sample(list, sample_size=1, gen_seed="No"):
    # Generate a random seed that is 8 characters
    seed = generate_seed(gen_seed=gen_seed, seed_length=8)

    # Set the seed
    random.seed(seed)

    # Take a sample of 10 items
    sample = random.sample(list, sample_size)

    return sample

# Method to randomly shuffle a list
def randomly_shuffle(list, gen_seed="No"):
    # Generate a random seed that is 8 characters
    seed = generate_seed(gen_seed=gen_seed, seed_length=8)

    # Set the seed
    random.seed(seed)

    # Shuffle the list using the seeded random
    random.shuffle(list)

    return list

# Method to divide up a list that is particularly large (for performance)
def divide_list_into_chunks(list, elements_per_list, gen_seed="No"):
    # Calculate how many lists can be created based on elements_per_list parameter
    num_of_lists = (len(list) // elements_per_list) + 1 # +1 for the remainder of the list
    # print(num_of_lists)

    # Shuffle list with seed
    shuffled_list = randomly_shuffle(list=list, gen_seed=gen_seed)

    return numpy.array_split(shuffled_list, num_of_lists)

# Method to load a corpus and divide it up into chunks; calls pipeline
def load_korean_corpus():
    input("\nWarning: Any corpora you load will be downloaded to your computer. This will take up a variable amount of storage (100s of MBs - 10s of Gbs). "
           "Also note, if you do not have the resulting files from running the program at least one time, the process of translating the different "
           "versions of the text takes a significant amount of time. If you wish to completely recreate the process of analyzing and translating "
           "the corpus I used (korean_hate_speech), then please be patient. \nPress Enter to continue.")

    # All corpora names from Korpora package (some do not work)
    for corpora in Korpora.corpus_list():
        print(corpora)
        # kcbert
        # korean_chatbot_data
        # korean_hate_speech
        # korean_parallel_koen_news
        # korean_petitions
        # kornli
        # korsts
        # kowikitext
        # namuwikitext
        # naver_changwon_ner
        # nsmc
        # question_pair
        # modu_news
        # modu_messenger
        # modu_mp
        # modu_ne
        # modu_spoken
        # modu_web
        # modu_written
        # open_subtitles
        # aihub_translation
        # aihub_spoken_translation
        # aihub_conversation_translation
        # aihub_news_translation
        # aihub_korean_culture_translation
        # aihub_decree_translation
        # aihub_government_website_translation

    corpus_name = str(input("Please enter the name of the corpus to load (see above): ") or "korean_hate_speech") # Default value :)
    try:
        # Load a specific corpus
        corpus = Korpora.load(f'{corpus_name}')
        print(f"Loading the {corpus_name} corpus...")

        # Access text as a list (comes originally as a tuple)
        if corpus_name == "korean_hate_speech":
            corpus_text = list(corpus.train.texts)
        else:
            print("Not all corpora from Korpora are handled the same way! Specific implementation should be added here. For now, just use the default value (korean_hate_speech)")
            return

        # Prompt user to regenerate the seed
        gen_seed = input("\nThis project uses the 'random' package for various operations. If you already have a 'seed.txt', do you want to generate a new seed? (Yes/No): ")
    
        # See if it worked by printing 5 elements from list
        print(f"\nPrinting 5 lines from the {corpus_name} corpus...")
        for line in randomly_sample(list=corpus_text, sample_size=5, gen_seed=gen_seed):
            print(line)
        
        # Separate out corpus into chunks for performance reasons
        max_lines_per_list = 1000
        print(f"Splitting up the {corpus_name} corpus into chunks of approximately {max_lines_per_list} lines each...")
        for index, chunk in enumerate(divide_list_into_chunks(list=corpus_text, elements_per_list=max_lines_per_list, gen_seed=gen_seed)):
            print("Converting chunks of the corpus to a single string with each element on a new line...")
            chunk_name = f"{corpus_name}_chunk_{index}"
            chunk_of_corpus_as_string = ""
            with alive_bar(total=len(chunk), title=f"{chunk_name}") as bar:
                for sentence in chunk:
                    chunk_of_corpus_as_string += "\n" + (sentence) # This new line character will appear in analysis, but is needed for translating
                    bar()
            start_pipeline(corpus_title=chunk_name, corpus_text=chunk_of_corpus_as_string, gen_seed=gen_seed)
    except Exception as e:
        print(e)

# Method to generate some plots with matplotlib
def generate_plots():
    class Language(Enum):
        ENGLISH = 0
        KOREAN = 1
    class Stopword(Enum):
        WITH = 0
        WITHOUT = 1
    class Metric(Enum):
        CHAR_COUNT = 0
        WORD_COUNT = 1
        NOUN_COUNT = 2
        VERB_COUNT = 3
        ADVERB_COUNT = 4
        ADJECTIVE_COUNT = 5
        PARTICLE_COUNT = 6

    def average_metric(language, stopword, metric):
        metric_list = []
        for chunk in overall_data[language][stopword]:
            metric_list.append(chunk[metric])
        return sum(metric_list) / len(metric_list)

    try:
        # Load existing data
        with open("./corpus_data.json", 'r') as existing_json:
            existing_data = json.load(existing_json)

            # Represents a list for English or Korean, then a list for with or without stopwords, each chunk, and then all metrics in their own list
            overall_data = []

            # Save the data for each language, for each version (stopwords), for each chunk, and for each metric
            for language in existing_data["overall_analysis"]:
                language_data = []
                for version in existing_data["overall_analysis"][language]:
                    version_data = []
                    for chunk in existing_data["overall_analysis"][language][version]:
                        chunk_data = []
                        for metric in existing_data['overall_analysis'][language][version][chunk]:
                            # We don't want to include the data for sampled_sentences for this
                            metric_value = existing_data['overall_analysis'][language][version][chunk][metric]
                            chunk_data.append(metric_value)
                        version_data.append(chunk_data)
                    language_data.append(version_data)
                overall_data.append(language_data)
            
            # Make a bar graph for every averaged metric
            for metric in Metric:
                # Load averaged metric data for each of our datasets
                chart_data = [average_metric(language=Language.KOREAN.value, stopword=Stopword.WITH.value, metric=metric.value), 
                              average_metric(language=Language.KOREAN.value, stopword=Stopword.WITHOUT.value, metric=metric.value),
                              average_metric(language=Language.ENGLISH.value, stopword=Stopword.WITH.value, metric=metric.value),
                              average_metric(language=Language.ENGLISH.value, stopword=Stopword.WITHOUT.value, metric=metric.value)]

                # Create a bar graph
                pyplot.bar(range(len(chart_data)), chart_data)

                # Add labels for each bar
                labels = ['Ko_W_Sw', 'Ko_Wo_Sw', 'En_W_Sw', 'En_Wo_Sw']
                pyplot.xticks(range(len(chart_data)), labels)

                # Add labels and title
                pyplot.xlabel('(Korean/English)_(With/Without)_Stopwords')
                pyplot.ylabel('Counts')
                pyplot.title(f"Average {metric.name} of 'korean_hate_speech' Corpus")

                # Save the plot
                pyplot.savefig(f"./Charts/{metric.name}_bargraph.png")

                # Reset the plot so data doesn't stack
                pyplot.clf()
            
            # Make a table for every averaged metric
            for metric in Metric:
                # Create column names
                columns = ['Language', 'Stopwords', metric.name]

                # Load appropriate data into a different list for the table
                data = [['Korean', 'With', average_metric(language=Language.KOREAN.value, stopword=Stopword.WITH.value, metric=metric.value)],
                        ['Korean', 'Without', average_metric(language=Language.KOREAN.value, stopword=Stopword.WITHOUT.value, metric=metric.value)],
                        ['English', 'With', average_metric(language=Language.ENGLISH.value, stopword=Stopword.WITH.value, metric=metric.value)],
                        ['English', 'Without', average_metric(language=Language.ENGLISH.value, stopword=Stopword.WITHOUT.value, metric=metric.value)]]

                # Create a figure and axis
                fig, ax = pyplot.subplots()

                # Hide the axes
                ax.axis('off')

                # Create a table
                pyplot.table(cellText=data, colLabels=columns, loc='center')

                # Save the table to a file
                pyplot.savefig(f"./Tables/{metric.name}_table.png")

            # Make a table for each chunk's sampled sentences
            columns = ['Corpus Chunk', 'Stopwords', 'Sampled Sentences', 'Ko Word Count', 
                        'Translated Sentences', 'En Word Count', 'Word Count Difference']
            
            # Assemble data in the correct format to match columns
            data = []
            for stopword in Stopword:
                print("\n")
                for i, chunk in enumerate(overall_data[Language.KOREAN.value][stopword.value]):
                    for j, sentence in enumerate(chunk[7]):
                        ko_sentence_word_count = len(sentence.split())
                        translated_sentence = overall_data[Language.ENGLISH.value][stopword.value][i][7][j]
                        en_sentence_word_count = len(translated_sentence.split())
                        word_count_diff = abs(ko_sentence_word_count - en_sentence_word_count)
                        data.append([f'korean_hate_speech_chunk_{i}', stopword.name, sentence, 
                                     ko_sentence_word_count, translated_sentence, en_sentence_word_count, word_count_diff])
                
            # Create CSV file
            with open('sampled_sentences_data.csv', 'w', encoding="utf-8", newline='') as csv_file:
                writer = csv.writer(csv_file)

                writer.writerow(columns)

                # Write each row in the list to the CSV
                for row in data:
                    writer.writerow(row)

                    # # Create a table
                    # table = pyplot.table(cellText=data, colLabels=columns, loc='center')

                    # fprop = fm.FontProperties(fname='NotoSansKR-Regular.ttf')

                    # for cell in table._cells:
                    #     table._cells[cell].set_text_props(fontproperties=fprop)

                    # # Hide the axes
                    # pyplot.axis('off')

                    # # Save the table to a file
                    # pyplot.savefig(f"./Tables/test_table_{i}_{stopword.name}.png", dpi=300)

                    # pyplot.clf()

            
            


    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist, tell the user to generate data first
        print(f"./corpus_data.json doesn't exist! This is created after performing analyses on a corpus.")

if __name__ == "__main__":
    create_folders(verbose=False) # Creates needed folders for organization if not already present
    # load_korean_corpus() # Starts the entire program
    generate_plots()