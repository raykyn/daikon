#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import io
import os

import numpy as np
import tensorflow as tf

from typing import List, Tuple

from daikon import vocab
from daikon import compgraph
from daikon import constants as C


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def load_vocabs(load_from: str) -> Tuple[vocab.Vocabulary, vocab.Vocabulary]:
    """
    """
    source_vocab = vocab.Vocabulary()
    source_vocab.load(os.path.join(load_from, C.SOURCE_VOCAB_FILENAME))
    target_vocab = vocab.Vocabulary()
    target_vocab.load(os.path.join(load_from, C.TARGET_VOCAB_FILENAME))

    return source_vocab, target_vocab


def translate_line(session: tf.Session,
                   line: str,
                   source_vocab: vocab.Vocabulary,
                   target_vocab: vocab.Vocabulary,
                   encoder_inputs: tf.Tensor,
                   decoder_inputs: tf.Tensor,
                   decoder_targets: tf.Tensor,
                   decoder_logits: tf.Tensor) -> str:
    """
    Translates one single input string.
    """

    source_ids = np.array(source_vocab.get_ids(line.split())).reshape(1, -1)

    #print(line)

    # instead of one list, we have a dictionary of float : list pairs
    # the float is always equal to the probability of this sentence
    #~ translated_ids = []  # type: List[int]
    
    # num of beams
    k = 5
    
    # new list of finished translations
    sent_dict = {}
    
    finished_sent_dict = {}

    for _ in range(C.TRANSLATION_MAX_LEN):

        # target ids will serve as decoder inputs and decoder targets,
        # but decoder targets will not be used to compute logits
        
        potential_sentences = {}
        
        if k == 0:
            break
        
        if len(sent_dict) == 0:
            target_ids = np.array([C.BOS_ID]).reshape(1, -1)
            
            feed_dict = {encoder_inputs: source_ids,
                         decoder_inputs: target_ids,
                         decoder_targets: target_ids}
            logits_result = session.run([decoder_logits], feed_dict=feed_dict)
            
            next_symbol_logits = softmax(logits_result[0][0][-1])
            
            potential_next_ids = []
            
            for __ in range(k):
                next_id = np.argmax(next_symbol_logits)
                next_id_value = next_symbol_logits[next_id]
                potential_next_ids.append((next_id, next_id_value))
                # after finding, we delete the element
                next_symbol_logits = np.delete(next_symbol_logits, next_id)
                
            #print("POTENTIAL NEXTS", potential_next_ids)
            #print(target_vocab.get_words([x[0] for x in potential_next_ids]))
               
            #print("POTENTIAL START", potential_next_ids)
 
            for new_id in potential_next_ids:
                if new_id not in [C.EOS_ID, C.PAD_ID]:
                    sent_dict[new_id[1]] = (new_id[0],)
            #print("START", sent_dict)
                
        else:
            for prob, sent in sent_dict.items():
                target_ids = np.array([C.BOS_ID] + list(sent)).reshape(1, -1)

                feed_dict = {encoder_inputs: source_ids,
                             decoder_inputs: target_ids,
                             decoder_targets: target_ids}
                logits_result = session.run([decoder_logits], feed_dict=feed_dict)

                # first session result, first item in batch, target symbol at last position
                next_symbol_logits = softmax(logits_result[0][0][-1])
                # 1. change:
                # retrieve k number of highest elements
                # loop argmax, everytime the highest has been found, delete it and argmax again
                # till k highest have been found.
                #~ next_id = np.argmax(next_symbol_logits)
                potential_next_ids = []
                
                for __ in range(k):
                    next_id = np.argmax(next_symbol_logits)
                    next_id_value = next_symbol_logits[next_id]
                    potential_next_ids.append((next_id, next_id_value))
                    # after finding, we delete the element
                    next_symbol_logits = np.delete(next_symbol_logits, next_id)
                    
                #print("POTENTIAL NEXTS", potential_next_ids)
                #print(target_vocab.get_words([x[0] for x in potential_next_ids]))
                    
                for new_id in potential_next_ids:
                    #print(sent)
                    #print(new_id)
                    new_sent = list(sent)
                    new_sent.append(new_id[0])
                    #print(new_sent)
                    new_value = prob * new_id[1]
                    potential_sentences[new_value] = new_sent
            
            # clear sent dict for the next loop
            sent_dict = {}
            # decide which k sentences are taken
            potential_sentences = sorted(potential_sentences.items(), reverse=True)[:k]
            #print("CHOSEN:", potential_sentences)
            for val, sent in potential_sentences:
                #print(sent)
                # if ending in <EOS>, add to finished
                if sent[-1] in [C.EOS_ID, C.PAD_ID]:
                    finished_sent_dict[val] = sent
                    k -= 1
                # else continue
                else:
                    sent_dict[val] = sent

            #print("CHOSEN", sent_dict)
                                
    # normalize the remaining sentences by length-alpha
    norm_dict = {}
    for val, sent in finished_sent_dict.items():
        if len(sent) > 0:
            val = np.log10(val) / len(sent)**0.65
            norm_dict[val] = sent
    
    #print("LEN_NORM", norm_dict)

    # only return our best translation
    try:
        best_sent = sorted(norm_dict.items(), reverse=True)[0][1]
    except:
        print("empty line...")
        print(norm_dict)
        best_sent = []    

    #print("BEST", best_sent)

    words = target_vocab.get_words(best_sent)

    #print("WORDS", words)

    return ' '.join(words)


def translate_lines(load_from: str,
                    input_lines: List[str],
                    train_mode: bool = False,
                    **kwargs) -> List[str]:
    """
    Translates a list of strings.
    """
    source_vocab, target_vocab = load_vocabs(load_from)

    # fix batch_size to 1
    encoder_inputs, decoder_targets, decoder_inputs, _, _, decoder_logits, _ = compgraph.define_computation_graph(source_vocab.size, target_vocab.size, 1)

    saver = tf.train.Saver()

    with tf.Session() as session:

        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        translations = []

        for line in input_lines:
            translation = translate_line(session, line, source_vocab, target_vocab, encoder_inputs, decoder_inputs, decoder_targets, decoder_logits)
            translations.append(translation)

    return translations


def translate_file(load_from: str, input_file_handle: io.TextIOWrapper, output_file_handle: io.TextIOWrapper, **kwargs):
    """
    Translates all lines that can be read from an open file handle. Translations
    are written directly to an output file handle.
    """
    source_vocab, target_vocab = load_vocabs(load_from)

    # fix batch_size to 1
    encoder_inputs, decoder_targets, decoder_inputs, _, _, decoder_logits, _ = compgraph.define_computation_graph(source_vocab.size, target_vocab.size, 1)

    saver = tf.train.Saver()

    with tf.Session() as session:

        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        for line in input_file_handle:
            translation = translate_line(session, line, source_vocab, target_vocab, encoder_inputs, decoder_inputs, decoder_targets, decoder_logits)
            output_file_handle.write(translation + "\n")
