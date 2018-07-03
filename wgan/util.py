import os
import logging
import numpy as np
import toml
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from glob import glob
import json
import pandas as pd
import re


def raise_error(condition, msg):
    if condition:
        raise ValueError(msg)


def create_log(out_file_path=None):
    """ Logging
        If `out_file_path` is None, only show in terminal
        or else save log file in `out_file_path`
    Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """
    handler = logging.StreamHandler()
    if out_file_path is not None:
        if os.path.exists(out_file_path):
            os.remove(out_file_path)
        handler = logging.FileHandler(out_file_path)

    logger = logging.getLogger(out_file_path)
    # avoid overlap logger
    if len(logger.handlers) > 0:
        return logger
    else:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


def checkpoint_version(config: dict,
                       path_to_checkpoints: str,
                       warm_start: bool=False,
                       progress_n: int=None):
    """ Checkpoint versioner
    :param config:
    :param path_to_checkpoints: `./checkpoint/lam
    :param warm_start: if False, create new directory for config, which not exist
    :param progress_n: progress checkpoint number.
    :return:
    """

    checkpoints = glob('%s/*' % path_to_checkpoints)

    # check if there are any checkpoints with same hyperparameters
    target_checkpoints = None
    for i in checkpoints:
        if len(glob('%s/*.ckpt.meta' % i)) == 0:
            continue
        json_dict = json.load(open('%s/hyperparameters.json' % i, 'r'))
        if config == json_dict:
            raise_error(target_checkpoints is not None, 'checkpoints are duplicated `%s`' % i)
            raise_error(not warm_start, 'checkpoints are already exist at `%s`' % i)
            target_checkpoints = i

    # if new, make new checkpoints directory, else return None if there are no checkpoints with given config
    if warm_start:
        raise_error(target_checkpoints is None, 'no checkpoints at `%s`' % path_to_checkpoints)
        # checkpoint/network/model-100.ckpt, meta-100.csv, hyperparameters.json
        # if version of progress not declared, choose biggest version in terms of epoch
        if progress_n is None:
            list_of_ckpt = glob('%s/*.ckpt.meta' % target_checkpoints)
            checkpoint_epoch = [int(re.sub(r'model-([\d]*).ckpt', r'\1', i.split('/')[-1])) for i in list_of_ckpt]
            progress_n = int(np.max(checkpoint_epoch))

        # get meta data
        train_info = pd.read_csv('%s/meta-%i.csv' % (target_checkpoints, progress_n), index_col=0).T
        learning_rate = train_info['learning_rate'].values[0]  # since learning rate can be decayed, use saved lr.
        ini_epoch = int(train_info['epoch'].values[0]) + 1
        target_model = '%s/model-%i.ckpt' % (target_checkpoints, progress_n)
        return dict(checkpoint_dir=target_checkpoints,
                    checkpoint=target_model,
                    ini_epoch=ini_epoch,
                    learning_rate=learning_rate)
    else:
        new_checkpoint_id = len(glob('%s/*' % path_to_checkpoints))
        new_checkpoint_path = '%s/%i' % (path_to_checkpoints, new_checkpoint_id)
        os.makedirs(new_checkpoint_path, exist_ok=True)
        with open('%s/hyperparameters.json' % new_checkpoint_path, 'w') as outfile:
            json.dump(config, outfile)
        return dict(checkpoint_dir=new_checkpoint_path,
                    checkpoint=None,
                    ini_epoch=0,
                    learning_rate=config['learning_rate'])


class BaseModel(object):

    def __init__(self,
                 config,
                 learning_rate,
                 batch_size,
                 gradient_clip,
                 layer_norm,
                 keep_prob,
                 keep_prob_r,
                 weight_decay,
                 optimizer,
                 load_lookup_word,
                 load_lookup_char,
                 load_lookup_output,
                 fine_tune_model,
                 fine_tune_model_binary,
                 crf,
                 character,
                 clean):

        self._logger = create_log()
        self._config = config
        self._lr = learning_rate
        self._clip = gradient_clip
        self._layer_norm = layer_norm
        self._keep_prob = keep_prob
        self._keep_prob_r = keep_prob_r
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._character = character  # if the model uses character
        self.__tokenizer = Tokenizer() if clean else None
        self.__crf = crf  # if the model uses crf

        # load lookup (character)
        self.__lookup_char = load_lookup(load_lookup_char)
        self._vocab_char = len(self.__lookup_char)
        # load lookup (output tag)
        self.__lookup_output = load_lookup(load_lookup_output)
        self._vocab_output = len(self.__lookup_output)
        # load lookup (word)
        self.__lookup_word = load_lookup(load_lookup_word)
        self._vocab_word = len(self.__lookup_word)
        self._embedding_size = len(list(self.__lookup_word.items())[0][1])
        # embedding model
        if fine_tune_model:
            self._fine_tune_model = gensim.models.KeyedVectors.load_word2vec_format(fine_tune_model,
                                                                                    binary=fine_tune_model_binary)
            if self._fine_tune_model.vector_size != self._embedding_size:
                raise ValueError('Size of embedding (%i) and lookup table (%i) is not same.'
                                 % (self._fine_tune_model.vector_size, self._embedding_size))
        else:
            self._fine_tune_model = None

    @staticmethod
    def train(epoch,
              model,
              learning_rate: float = None,
              checkpoint="./",
              warm_start: bool = True,
              debug: bool = True,
              progress_interval: int = 10
              ):

        """ Train model."""

        # logger
        logger = create_log() if debug else None

        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint, exist_ok=True)

        log("epoch size (%i),  (train: %i, valid: %i, test: %i), batch size (%i), variables (%i)" %
                    (epoch, feeder_train.data_size, feeder_valid.data_size,
                     0 if feeder_test is None else feeder_test.data_size,
                     self._batch_size, self.total_variable_number),
            logger)

        if ini_epoch != 0:
            epoch += ini_epoch
            logger.info('- train from epoch %i to %i' % (ini_epoch, epoch))

        result_valid, result_train = [], []

        # learning rate scheduler
        current_lr = self._lr if ini_learning_rate is None else ini_learning_rate
        scheduler = LearningRateScheduler(current_lr)

        _e = 0
        for _e in range(ini_epoch, epoch):

            # Train
            t_re_macro, t_re_micro, t_loss = self.__train_epoch(feeder_train, True, verbose, logger, lr=current_lr)
            result_train.append(t_re_macro + t_re_micro + [t_loss])

            v_re_macro, v_re_micro, v_loss = self.__train_epoch(feeder_valid, False, False, logger)
            result_valid.append(v_re_macro + v_re_micro + [v_loss])

            # log
            msg = 'e %i ' % _e
            msg += "[train] l:%.2f, f-mac:%.2f, f-mic:%.2f " % (t_loss, t_re_macro[2], t_re_micro[2])
            msg += "[valid] l:%.2f, f-mac:%.2f, f-mic:%.2f" % (v_loss, v_re_macro[2], v_re_micro[2])
            msg += " [lr] %0.3f" % current_lr
            logger.info(msg)

            # Save progress model
            if _e % 50 == 0 and not _e == 0:
                self._saver.save(self._sess, "%s/progress_%i_model.ckpt" % (save_path, _e))
                pd.DataFrame([_e, current_lr], index=['epoch', 'learning_rate']).to_csv(
                    '%s/progress_%i_train_info.csv' % (save_path, _e))

            current_lr, stop_flg = scheduler(t_loss)
            if stop_flg:
                logger.info('[Early stop] training is stopped by scheduler. ')
                break

        # Test
        if feeder_test is not None:
            re_macro, re_micro, loss = self.__train_epoch(feeder_test, False, False, logger)
            logger.info("[test] l:%.2f, f-mac:%.2f, f-mic:%.2f" % (loss, re_macro[2], re_micro[2]))

        # Save
        logger.info('Saving ....')
        self._saver.save(self._sess, "%s/model.ckpt" % save_path)
        pd.DataFrame([_e, current_lr], index=['epoch', 'learning_rate']).to_csv('%s/train_info.csv' % save_path)
        np.savez("%s/error_%i_%i.npz" % (save_path, ini_epoch, ini_epoch + _e), train=result_train, valid=result_valid)

        log('saved at %s' % checkpoint, logger)

    def __train_epoch(self, feeder, is_train, verbose, logger, lr=None):
        loss, prediction, true = 0.0, [], []
        for step, (inputs, outputs) in enumerate(feeder):
            (word, word_len), (char, char_len) = self.__process_input(inputs)
            out, length = self.__process_output(outputs)
            if np.sum(np.array(length) != np.array(word_len)) != 0:
                raise ValueError('Error of batch feeder. Length conflict !')

            fd = {self.input_word: word, self.seq_len_word: word_len, self.output: out, self.is_train: is_train}
            fv = [self._loss, self._logit]
            if self._character:
                fd[self.input_char] = char
                fd[self.seq_len_char] = char_len
            if self.__crf:
                fv.append(self._transition_params)

            if is_train:
                if lr is None:
                    raise ValueError('Learning rate is not given.')
                fd[self.learning_rate] = lr
                fv.append(self._train_op)
            result = self._sess.run(fv, feed_dict=fd)
            loss += result[0]

            if self.__crf:  # iterate over the sentences because no batching in viterbi_decode
                for __logit, _seq_len in zip(result[1], word_len):  # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(__logit[:int(_seq_len), :], result[2])
                    prediction.append(viterbi_seq)
            else:
                prediction.extend([np.argmax(__logit[:int(_word_len), :], axis=1).astype(int)
                                   for __logit, _word_len in zip(result[1], word_len)])
            true.extend([o[0:int(word_len[ind])] for ind, o in enumerate(out)])

            if verbose and step % (feeder.iteration_number // 10) == 10:
                _, _, f_mac, _ = f_score(prediction[-len(out):], out, 'macro')
                _, _, f_mic, _ = f_score(prediction[-len(out):], out, 'micro')
                logger.info("%i/%i l: %.2f, f-mac: %.2f, f-mic: %.2f"
                            % (step, feeder.iteration_number, loss / step, f_mac, f_mic))

        re_macro = f_score(prediction, true, 'macro')  # precision, recall, f1, acc
        re_micro = f_score(prediction, true, 'micro')
        loss = loss / feeder.iteration_number
        return list(re_macro), list(re_micro), loss

    def predict(self, data_input, return_probability=False):
        """ Prediction based on trained model.

        :param list data_input: list of target sentences
        :return:
        """

        inv_dict = {v: k for k, v in self.__lookup_output.items()}

        if len(self.__lookup_word) == 0:
            raise ValueError("Train model before prediction!!")

        (word, word_len), (char, char_len) = self.__process_input(data_input)
        fd = {self.input_word: word, self.seq_len_word: word_len, self.is_train: False}
        if self._character:
            fd[self.input_char] = char
            fd[self.seq_len_char] = char_len
        if self.__crf:
            # get tag scores and transition params of CRF
            _logit, _trans = self._sess.run([self._logit, self._transition_params], feed_dict=fd)
            # iterate over the sentences because no batching in viterbi_decode
            prediction = []
            for __logit, _word_len in zip(_logit, word_len):  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(__logit[:int(_word_len), :], _trans)
                prediction.append(' '.join([inv_dict[_tmp] for _tmp in viterbi_seq]))
        else:
            _logit = self._sess.run([self._logit], feed_dict=fd)
            prediction = np.array([np.argmax(__logit[:int(_word_len), :], axis=1).astype(int)
                                  for __logit, _word_len in zip(_logit, word_len)])
        if return_probability:

            prob = [self.softmax(__logit[:int(_word_len), :]) for __logit, _word_len in zip(_logit, word_len)]
            return prediction, (prob, inv_dict)
        else:
            return prediction

    def probability(self, data_input):
        """

        :param list data_input: list of target sentences
        :return list of array, which contains probability over tags and look up table for tag
        """

        inv_dict = {v: k for k, v in self.__lookup_output.items()}

        (word, word_len), (char, char_len) = self.__process_input(data_input)
        fd = {self.input_word: word, self.seq_len_word: word_len, self.is_train: False}
        if self._character:
            fd[self.input_char] = char
            fd[self.seq_len_char] = char_len
        _logit = self._sess.run([self._logit], feed_dict=fd)
        prob = [self.softmax(__logit[:int(_word_len), :]) for __logit, _word_len in zip(_logit, word_len)]
        return (prob, inv_dict)

    def __process_output(self, batch_data):
        """ padding, indexing the tags """
        tag, length = [], []
        for _x, text in enumerate(batch_data):
            _tmp = np.zeros(self._config["max_len_word"])
            tokens = text.split(' ')
            tokens = tokens[0:int(np.min([len(tokens), self._config["max_len_word"]]))]
            length.append(len(tokens))
            _tmp[0:len(tokens)] = [self.__lookup_output[t] if t in self.__lookup_output.keys() else 0 for t in tokens]
            tag.append(_tmp)
        return np.array(tag), np.array(length)

    def __process_input(self, batch_data):
        """ Padding and embedding """
        output_w, output_c, seq_len_w, seq_len_c = [], [], [], []
        for _x, text in enumerate(batch_data):

            # process word
            _tmp = np.zeros((self._config["max_len_word"], self._embedding_size))
            if self.__tokenizer is not None:
                tokens = self.__tokenizer(text, keep_length=True).split(' ')
                identifier = self.__tokenizer.unknown_identifier
            else:
                tokens = text.split(' ')
                identifier = []
            tokens = tokens[0:int(np.min([len(tokens), self._config["max_len_word"]]))]  # cut off long-length sentence
            seq_len_w.append(len(tokens))  # store actual length
            _tmp[0:len(tokens)] = np.array([self.__embedding(t) for t in tokens])
            output_w.append(_tmp)

            # process characters
            if self._character:
                _tmp_c = np.zeros((self._config["max_len_word"], self._config["max_len_char"]))
                _tmp_c_seq = np.zeros(self._config["max_len_word"])
                for _y, token in enumerate(tokens):
                    if token in identifier:  # put character id `0` with length `1` if identifier
                        _tmp_c[_y, 0] = 1
                        _tmp_c_seq[_y] = 1
                    else:
                        token = token[0:int(np.min([len(token), self._config["max_len_char"]]))]  # cut off
                        _tmp_c[_y, 0:len(token)] = np.array([self.__char(_c) for _c in token])
                        _tmp_c_seq[_y] = len(token)
                seq_len_c.append(_tmp_c_seq)
                output_c.append(_tmp_c)
        return (np.array(output_w), np.array(seq_len_w)), (np.array(output_c), np.array(seq_len_c))

    def __char(self, char):
        """ Project character to id, padding by 0"""
        return 0 if char not in self.__lookup_char.keys() else self.__lookup_char[char]

    def __embedding(self, token):
        """ Fine-tuned embedding """
        if token not in self.__lookup_word.keys():
            if self._fine_tune_model is not None:
                try:
                    _tmp = self._fine_tune_model[token]
                except KeyError:
                    _tmp = np.zeros(self._embedding_size)
            else:  # if no embedding, lookup table is never updated
                _tmp = np.zeros(self._embedding_size)
        else:
            _tmp = self.__lookup_word[token]
        return _tmp

    @staticmethod
    def softmax(logit):
        """logit: (seq, output_dimension) -> prob: same shape"""
        eps = 1e-5
        exp_logit = np.exp(logit)
        norm = np.sum(exp_logit, axis=1)
        return exp_logit / np.expand_dims(norm + eps, axis=1)

    @property
    def batch_size(self):
        return self._batch_size



