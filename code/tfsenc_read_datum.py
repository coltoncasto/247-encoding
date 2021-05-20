import os
import pickle

import numpy as np
import pandas as pd
from collections import Counter
import gensim.downloader as api
from transformers import GPT2Tokenizer
from transformers.utils import dummy_flax_objects


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        dict/list: pickle contents returned as dataframe
    """
    with open(file, 'rb') as fh:
        datum = pickle.load(fh)

    return datum


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding
    """
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())
    df = df[~df['is_nan']]

    return df


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def gen_word2vec_embeddings(df):
    glove = api.load('glove-wiki-gigaword-50')
    df['pred_embeddings'] = df['top1_pred'].apply(lambda x: get_vector(x, glove))
    df = df[~df['pred_embeddings'].isna()] # remove any empty embeddings
    return df


def return_stitch_index(args):
    """[summary]
    Args:
        args ([type]): [description]
    Returns:
        [type]: [description]
    """
    stitch_file = os.path.join(args.PICKLE_DIR, str(args.sid), args.stitch_file)
    stitch_index = load_pickle(stitch_file)
    return stitch_index


def adjust_onset_offset(args, df):
    """[summary]
    Args:
        args ([type]): [description]
        df ([type]): [description]
    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)
    assert len(stitch_index) == df.conversation_id.nunique()

    stitch_index = [0] + stitch_index[:-1]

    df['adjusted_onset'], df['onset'] = df['onset'], np.nan
    df['adjusted_offset'], df['offset'] = df['offset'], np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        shift = stitch_index[idx]
        df.loc[df.conversation_id == conv,
               'onset'] = df.loc[df.conversation_id == conv,
                                 'adjusted_onset'] - shift
        df.loc[df.conversation_id == conv,
               'offset'] = df.loc[df.conversation_id == conv,
                                  'adjusted_offset'] - shift

    return df


def add_signal_length(args, df):
    """[summary]
    Args:
        args ([type]): [description]
        df ([type]): [description]
    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)

    signal_lengths = np.diff(stitch_index).tolist()
    signal_lengths.insert(0, stitch_index[0])

    df['conv_signal_length'] = np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        df.loc[df.conversation_id == conv,
               'conv_signal_length'] = signal_lengths[idx]

    return df


def get_most_frequent(df):
    """
    """
    word_counts = Counter(df.token)
    third_index = round(len(word_counts)/2)
    word_list = word_counts.most_common(third_index)
    word_list = [x[0] for x in word_list]

    # get least frequent length
    reverse_order = sorted(word_counts.items(), key=lambda x:x[1])
    sum_least = sum([x[1] for x in reverse_order[:third_index]])
    print(f'Least frequency size: {sum_least}')

    # randomly samply to match the size of the least frequent words
    df_new = pd.DataFrame()
    df = df.sample(n=len(df), random_state=111)
    for row in df.iterrows():
        if row[1].token in word_list:
            df_new = df_new.append(dict(row[1]), ignore_index=True)
    
        if len(df_new)==sum_least: 
            break

    df_new['conversation_id'] = [int(x) for x in df_new['conversation_id']]
    print(df_new.shape)
    return df_new


def get_least_frequent(df):
    """
    """
    word_counts = Counter(df.token)

    # get least frequent third
    reverse_order = sorted(word_counts.items(), key=lambda x:x[1])
    index = round(len(reverse_order)/2)
    word_list = [x[0] for x in reverse_order[:index]]

    df_new = pd.DataFrame()
    for row in df.iterrows():
        if row[1].token in word_list:
            df_new = df_new.append(dict(row[1]), ignore_index=True)

    # df_new = df_new.sort_values(by = 'adjusted')
    df_new['conversation_id'] = [int(x) for x in df_new['conversation_id']]
    print(df_new.shape)
    return df_new


def add_conversation_onset_offset(df):
    convo_onsets = np.zeros((len(df),1))
    convo_offsets = np.zeros((len(df),1))
    total_len = 0
    for i in range(0,df.conversation_id.values[-1]):
        section = df.conversation_id == i+1
        convo_onsets[section] = total_len
        if sum(section) == 0:
            print(f'skipped {i+1}')
            total_len += 10667 # length of convo 61
            continue

        total_len += df[section].conv_signal_length.values[0]
        convo_offsets[section] = total_len - 1
    
    print(total_len)
    convo_onsets = np.array([int(onset) for onset in convo_onsets])
    convo_offsets = np.array([int(offset) for offset in convo_offsets])

    df['conversation_onset'] = convo_onsets
    df['conversation_offset'] = convo_offsets
    return df


def read_datum(args):
    """Read and process the datum based on input arguments

    Args:
        args (namespace): commandline arguments

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """
    file_name = os.path.join(args.PICKLE_DIR, str(args.sid),
                             args.load_emb_file)
    datum = load_pickle(file_name)

    df = pd.DataFrame.from_dict(datum)
    df = add_signal_length(args, df)
    df = adjust_onset_offset(args, df)
    df = drop_nan_embeddings(df)
    df = add_conversation_onset_offset(df)

    # only look at one conversation if requested
    if args.conversation_id_flag:
        df = df[df['conversation_id'] == args.conversation_id]
        print(f'Conversation {args.conversation_id} has {len(df)} words')

    # use columns where token is root
    # TODO - note that this does not work with "gpt2-xl"
    if 'gpt2-xl' in [args.align_with, args.emb_type]:
        df = df[df['gpt2-xl_token_is_root']]
    elif 'bert' in [args.align_with, args.emb_type]:
        df = df[df['bert_token_is_root']]
    else:
        pass
    

    print(df.shape)
    # remove instances with no glove embeddings
    if args.remove_glove:
        df = df[~df['glove50_embeddings'].isna()]
    print(df.shape)

    # if encoding is on glove embeddings copy them into 'embeddings' column
    if args.emb_type == 'glove50':
        df['embeddings'] = df['glove50_embeddings']

    # format predictions and tokens for comparison
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl',
                                          add_prefix_space=True)
    df['token'] = [tokenizer.convert_tokens_to_string(token).strip() 
                        for token in df['token']]
    df['top1_pred'] = df['top1_pred'].str.strip()

    if args.split_flag:
        if 'correct' == args.split_by:
            df = df[df.token == df.top1_pred]
        elif 'incorrect' in args.split_by:
            df = df[df.token != df.top1_pred]

            # uncomment for encoding - makes pred/actual same group
            # df = gen_word2vec_embeddings(df)

            # replace embeddings of actual with glove of top1_pred
            if 'predicted' in args.split_by:
                df['embeddings'] = df['pred_embeddings']

            # uncomment for encoding - makes pred/actual same group
            # df = df.drop(columns=['pred_embeddings'])

        # for erp on most/least frequent words
        if 'frequent' in args.split_by:
            
            if 'most' in args.split_by:
                df = get_most_frequent(df)
            else: 
                df = get_least_frequent(df)
        
    print(df.shape)
    return df
