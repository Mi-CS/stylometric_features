import numpy as np
import re
import scipy.stats as stat
import nltk
from collections import Counter


ALLOWED_SPECIAL_CHARS = r"(),.:;-–!?_'" + '"'
POSSIBLE_TAGS = "CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP TO UH VB VBG VBD VBN VBP VBZ WDT WP WP$ WRB . , : ( ) ''".split()


def get_stylographic_feat(tokens_list): 
    return np.concatenate((_lexical_features(tokens_list),
                          _syntactic_features(tokens_list)))


def _lexical_features(list_of_tokens): 
    words_list = []
    sp_char_list = []
    for token in list_of_tokens: 
        if token in ALLOWED_SPECIAL_CHARS:
            sp_char_list.append(token)
        else:
            words_list.append(token)
    
    n_tokens = len(list_of_tokens)
    n_words = len(words_list)
    v_words = len(set(words_list))
    word_lengths = np.array([len(word) for word in words_list])
    avg_word_lengths = word_lengths.mean() # Average word length
    std_word_lengths = word_lengths.std() # Standard deviation of word length
    v_n_words = v_words/n_words # % distinct words in words
    c_unique = Counter(words_list) # Counter of unique words
    i_vi = Counter(c_unique.values()) # i_vi[i_times] = V_i 
    VR_K = 10**4 * sum([i**2 * i_vi[i] - n_words for i in i_vi]) / n_words**2
    VR_R = v_words/np.sqrt(n_words)
    VR_C = np.log(v_words) / np.log(n_words)
    VR_H = (100 * np.log(n_words)) / ((1-i_vi[1])/v_words)
    VR_S = i_vi[2] / v_words
    VR_k = np.log(v_words) / np.log(np.log(n_words))
    VR_LN = (1-v_words**2) / (v_words**2 * np.log(n_words))
    entr = stat.entropy(list(c_unique.values()))
    
    full_string = " ".join(list_of_tokens)
    n_chars = len(full_string)
    freq_alpha = len(re.findall(r"[a-z]", full_string))/n_chars
    freq_sp = 1 - freq_alpha
    freq_common_punct = len(re.findall(r"[\.,;]", full_string))/n_chars
    
    return np.array([n_words, v_words, avg_word_lengths, std_word_lengths,
                     v_n_words, VR_K, VR_R, VR_C, 
                     VR_H, VR_S, VR_k, VR_LN,
                     entr, n_chars, freq_alpha, freq_sp, freq_common_punct])


# nltk tags:
# https://www.guru99.com/pos-tagging-chunking-nltk.html

def _syntactic_features(list_of_tokens): 
    n_tokens = len(list_of_tokens)
    tokens_tag = nltk.pos_tag(list_of_tokens)
    tag_dict = dict(zip(POSSIBLE_TAGS, [0]*len(POSSIBLE_TAGS)))
    for _, tag in tokens_tag: 
        if tag in tag_dict:
            tag_dict[tag] += 1
        else:
            pass
    
    # Normalize
    for token in tag_dict: 
        tag_dict[token] /= n_tokens
            
    noun_freq = tag_dict["NN"] + tag_dict["NNS"]
    prop_noun_freq = tag_dict["NNP"] + tag_dict["NNPS"]
    pronoun_freq = tag_dict["PRP"] + tag_dict["PRP$"]
    ordinal_adj_freq = tag_dict["JJ"]
    comp_adj_freq = tag_dict["JJR"]
    super_adj_freq = tag_dict["JJS"]
    adverb_freq = tag_dict['RB']
    comp_adverb_freq = tag_dict['RBR'] 
    super_adverb_freq = tag_dict['RBS']
    modal_freq = tag_dict['MD']
    base_verb_freq = tag_dict['VB'] 
    present_verb_freq = tag_dict['VBP'] + tag_dict['VBZ']
    past_verb_freq = tag_dict['VBD']
    pres_part_verb_freq = tag_dict['VBG']
    particles_freq = tag_dict['RP']
    wh_freq = tag_dict['WDT'] + tag_dict["WP"] + tag_dict["WP$"] + tag_dict['WRB']
    conj_freq = tag_dict['CC']
    num_freq = tag_dict["CD"]
    det_freq = tag_dict["DT"] + tag_dict["PDT"]
    ex_freq = tag_dict["EX"]
    to_freq = tag_dict["TO"]
    prepos_freq = tag_dict["IN"]
    genitive_freq = tag_dict["POS"]
    quot_dot_freq = tag_dict["."]
    commas_freq = tag_dict[","]
    sep_freq = tag_dict[":"]
    
    return np.array([noun_freq,
    prop_noun_freq,
    pronoun_freq,
    ordinal_adj_freq,
    comp_adj_freq, 
    super_adj_freq,
    adverb_freq, 
    comp_adverb_freq,
    super_adverb_freq, 
    modal_freq, 
    base_verb_freq, 
    present_verb_freq, 
    past_verb_freq,
    pres_part_verb_freq,
    particles_freq,
    wh_freq, 
    conj_freq, 
    num_freq,
    det_freq,
    ex_freq,
    to_freq ,
    prepos_freq ,
    genitive_freq,
    quot_dot_freq, 
    commas_freq,
    sep_freq])