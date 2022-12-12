'''
Data Loader
'''

import os
cwd = os.getcwd()
BOW_PATH = cwd + "/Data/ap.dat"
VOCAB_PATH = cwd + "/Data/vocab.txt"


# Loads the AP article dataset
def load_data():
    # Load index-to-word mapping
    print('Loading index-to-word mapping...')

    with open(VOCAB_PATH, 'r') as fh:
        raw_lines = fh.readlines()

    idx_to_words = [word.strip() for word in raw_lines]
    V = len(idx_to_words)

    # Load article BoW representations
    print('Loading article bag-of-word representations...')

    with open(BOW_PATH, 'r') as fh:
        raw_lines = fh.readlines()
        N = len(raw_lines)
        print('{} articles found.'.format(N))

    articles = np.zeros((N,V))
    nonzero_idxs = []

    # Process each article
    for i in tqdm(range(N)):
        split = raw_lines[i].split(' ')
        n_words = int(split[0]) # Number of words in the article
        split = split[1:] # BoW representations

        article = np.zeros((V,)) # Sparse V-vector
        nonzero_idx = [] # List of indices that have non-zero counts

        for bow in split:
            bow = bow.strip()
            word_idx, count = bow.split(':')

            nonzero_idx.append(int(word_idx))
            article[int(word_idx)] = count

        # Check if article words parsed correctly
        try:
            assert(len(nonzero_idx) == n_words)
        except:
            raise AssertionError('{}, {}'.format(len(nonzero_idx), n_words))

        articles[i] = article
        nonzero_idxs.append(sorted(nonzero_idx))

    return idx_to_words, articles, nonzero_idxs
