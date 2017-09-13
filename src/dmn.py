import numpy as np
import pandas as pd
import tensorflow as tf
import time
import re
import copy
import os

class dmn_plus:
    def __init__(self, task):
        self.vocab_size = 400000
        self.embedding_dim = 50
        self.hidden_layer_size = 80
        self.num_steps = 3
        self.batch_size = 100
        self.dropout_probability = 0.9
        self.l2_regularization_lambda = 0.001
        self.learning_rate = 0.001
        self.num_epochs = 128
        self.num_epochs_before_checking_valid_loss = 5
        self.num_consecutive_strips_before_stopping = 10
        self.datatype_id_dict = {'train': 0, 'valid': 1, 'test': 2, 'user': 3}
        self.datatypes = ['train', 'valid', 'test', 'user']
        self.load_embeddings()
        self.task = task
        self.load_data(self.task)
        self.create_tensorflow_graph()
        
    def load_embeddings(self):
        file = open("../../../datasets/glove_6b/glove.6B.50d.txt")    
        self.embedding = np.ndarray([self.vocab_size, self.embedding_dim])
        self.word_id_dict = {}
        self.id_word_dict = {}
        id = 0
        for line in file:
            items = line.split(' ')
            self.word_id_dict[items[0]] = id
            self.id_word_dict[id] = items[0]
            self.embedding[id,:] = np.array([float(i) for i in items[1:]])
            id += 1
        file.close()
        
    def load_babi_data(self, datatype_id):
        path_to_file_directory = "../../../datasets/facebook_babi/tasks_1-20_v1-2/en-valid-10k/"
        path_to_file = path_to_file_directory + 'qa' + str(self.task) + '_' + self.datatypes[datatype_id] + '.txt'
        
        file = open(path_to_file)
        num_words_in_longest_input_sentence = 0
        num_words_in_longest_question = 0
        num_sentences_in_each_chapter = []
        chapter_input = []
        data = []

        for line in file:
            items = re.sub('[?.]', '', line).lower().split()
            if items[-1].isdigit():
                # find the index of the second digit in that line
                index_of_second_digit = len(items)
                for index in range(len(items)-1, 0, -1):
                    if items[index].isdigit():
                        index_of_second_digit = index
                data.append({'I': copy.deepcopy(chapter_input),
                         'Q': items[1:index_of_second_digit-1],
                         'A': [items[index_of_second_digit-1]]})
                num_sentences_in_each_chapter.append(len(chapter_input))
                num_words_in_longest_question = max(num_words_in_longest_question, len(items[1:index_of_second_digit-1]))
            else:
                if items[0] == '1':
                    chapter_input = [items[1:]]
                else:
                    chapter_input.append(items[1:])
                num_words_in_longest_input_sentence = max(num_words_in_longest_input_sentence, len(items[1:]))
        file.close()

        num_sentences_in_longest_input = max(num_sentences_in_each_chapter)
        num_chapters = len(data)

        return([data, num_sentences_in_each_chapter, num_words_in_longest_input_sentence,
              num_words_in_longest_question, num_sentences_in_longest_input, num_chapters])
    
    def embed_and_pad_data(self, datatype_id):
        num_chapters = self.data_and_metadata[datatype_id][5]
        data_inputs = np.zeros([num_chapters, self.num_sentences_in_longest_input, self.num_words_in_longest_input_sentence, self.embedding_dim])
        data_questions = np.zeros([num_chapters, self.num_words_in_longest_question, self.embedding_dim])
        data_answers = np.zeros([num_chapters])
        for chapter_index, chapter in enumerate(self.data_and_metadata[datatype_id][0]):
            for sentence_index, sentence in enumerate(chapter['I']):
                data_inputs[chapter_index, sentence_index, 0:len(sentence), :] = self.embedding[[self.word_id_dict[word] for word in sentence]]
            data_questions[chapter_index, 0:len(chapter['Q']), :] = self.embedding[[self.word_id_dict[word] for word in chapter['Q']]]
            data_answers[chapter_index] = None if chapter['A'][0] == None else self.word_id_dict[chapter['A'][0]]
            
        return([data_inputs, data_questions, data_answers])
    
    def create_position_encoding(self):
        self.position_encoding = np.ones([self.embedding_dim, self.num_words_in_longest_input_sentence], dtype=np.float32)

        ## Below (my implementation, from section 3.1 in https://arxiv.org/pdf/1603.01417.pdf) didn't work.
        # for j in range(1, num_words_in_longest_input_sentence+1):
        #     for d in range(1, embedding_dim+1):
        #         position_encoding[d-1, j-1] = (1 - j/num_words_in_longest_input_sentence) - (d/embedding_dim)*(1 - 2*j/num_words_in_longest_input_sentence)

        ## Copied from https://github.com/domluna/memn2n
        ls = self.num_words_in_longest_input_sentence+1
        le = self.embedding_dim+1
        for i in range(1, le):
            for j in range(1, ls):
                self.position_encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        self.position_encoding = 1 + 4 * self.position_encoding / self.embedding_dim / self.num_words_in_longest_input_sentence
        self.position_encoding = np.transpose(self.position_encoding)
        
    def load_data(self, task):
        self.data_and_metadata = [self.load_babi_data(datatype_id = i) for i in range(3)]      
        self.data_and_metadata.append(self.preprocess_user_data())
        
        self.num_words_in_longest_input_sentence = max([self.data_and_metadata[i][2] for i in range(3)])
        self.num_words_in_longest_question = max([self.data_and_metadata[i][3] for i in range(3)])
        self.num_sentences_in_longest_input = max([self.data_and_metadata[i][4] for i in range(3)])
        
        self.embedded_data = [self.embed_and_pad_data(datatype_id = i) for i in range(4)]
        
        self.create_position_encoding()
        
    def answer_user_data(self, inputs, questions):    
        # load model on user data
        self.data_and_metadata[self.datatype_id_dict['user']] = self.preprocess_user_data(inputs, questions)
        self.embedded_data[self.datatype_id_dict['user']] = self.embed_and_pad_data(self.datatype_id_dict['user'])
        # get predictions
        predictions = self.sess.run(self.predictions, feed_dict = self.get_batch(datatype = 'user', batch_number = 0))
        predictions = [self.id_word_dict[id] for id in predictions]
        return(predictions)
    
    def preprocess_user_data(self, inputs = [], questions = []):    
        num_sentences_in_each_chapter = []
        chapter_input = []
        data = []

        for index in range(len(inputs)):
            input = re.sub('[?]', '', inputs[index]).lower().split('.')
            chapter_input = []
            for sentence in input:
                if sentence != '':
                    chapter_input.append(sentence.split())
            cleaned_question = re.sub('[?]', '', questions[index]).lower().split()
            data.append({'I': chapter_input,
                         'Q': cleaned_question,
                         'A': [None]})
            num_sentences_in_each_chapter.append(len(chapter_input))
        num_chapters = len(data)
        
        return([data, num_sentences_in_each_chapter, None,
              None, None, num_chapters])
        
    def get_batch(self, datatype, batch_number):
        index = self.datatype_id_dict[datatype]
        return {self.inputs: self.embedded_data[index][0][batch_number*self.batch_size: (batch_number+1)*self.batch_size],
                self.questions: self.embedded_data[index][1][batch_number*self.batch_size: (batch_number+1)*self.batch_size],
                self.answers: self.embedded_data[index][2][batch_number*self.batch_size: (batch_number+1)*self.batch_size],
                self.input_lengths: self.data_and_metadata[index][1][batch_number*self.batch_size: (batch_number+1)*self.batch_size]
               }
    
    def perform_epoch(self, num_chapters, datatype):
        epoch_loss = epoch_num_correct = 0
        for batch_idx in range(num_chapters/self.batch_size):
            batch_loss, batch_num_correct, _ = self.sess.run((self.loss, self.num_correct, (self.optimizer if datatype == 'train' else self.question_vector)), 
                                                                        feed_dict = self.get_batch(datatype = datatype, batch_number = batch_idx))
            epoch_loss += batch_loss
            epoch_num_correct += batch_num_correct
        return(epoch_loss, epoch_num_correct)
                
    def train(self):            
        self.sess.run(tf.global_variables_initializer())
        best_valid_loss = float("inf")
        previous_valid_loss = float("inf")
        is_valid_loss_greater_strip = []
        train_loss = []
        train_num_chapters = self.data_and_metadata[self.datatype_id_dict['train']][5]
        valid_num_chapters = self.data_and_metadata[self.datatype_id_dict['valid']][5]
        start_time = time.time()
        for epoch in range(self.num_epochs):
            epoch_loss, epoch_num_correct = self.perform_epoch(train_num_chapters, 'train')
            print("Epoch %d: %.2f%% complete, %d mins, Avg loss: %.2f, Num correct: %d, Accuracy: %.2f%%" % (epoch, 
                                                                                   epoch*100.0/self.num_epochs,
                                                                                    (time.time() - start_time)/60,
                                                                                   epoch_loss/train_num_chapters, 
                                                                                    epoch_num_correct,
                                                                                    epoch_num_correct*100.0/train_num_chapters))
            train_loss.append(epoch_loss/train_num_chapters)
            # early stopping
            if epoch%self.num_epochs_before_checking_valid_loss == 0:
                epoch_loss, epoch_num_correct = self.perform_epoch(valid_num_chapters, 'valid')
                print("\nValidation avg loss: %.2f, Num correct: %d, Accuracy: %.2f%%"%(epoch_loss/valid_num_chapters, epoch_num_correct, float(epoch_num_correct*100)/valid_num_chapters))
                # self.save()
                
                # UP stopping criterion (from http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)
                is_valid_loss_greater_strip.append(False if epoch_loss < previous_valid_loss else True)
                if(sum(is_valid_loss_greater_strip[-self.num_consecutive_strips_before_stopping:]) == self.num_consecutive_strips_before_stopping):
                    print("Stopping Early\nDuration: %d mins" % int((time.time() - start_time)/60))
                    return
                
                if epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                    self.save()                
                previous_valid_loss = epoch_loss
                
            # early stopping if progress is < 0.1
            progress = np.mean(train_loss[-5:])/min(train_loss[-5:]) - 1
            print(progress)
            if progress < 0.01 and epoch >= 24:
                if epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                    self.save()
                return
                # crude stopping criterion (stop as soon as validation loss increases)
                # if new validation loss is lower than the old one, save new weights
#                 if epoch_loss < old_valid_loss:
#                     old_valid_loss = epoch_loss
#                     self.save()
#                 # else, stop training
#                 else:
#                     print("Stopping Early\nDuration: %d mins" % int((time.time() - start_time)/60))
#                     return
        print("Duration: %d mins" % int((time.time() - start_time)/60))
        
    def save(self):
        # create filename based on task
        save_path = self.saver.save(self.sess, "../saved-models/task-%d.ckpt"%self.task)
        print("Model saved in file: %s\n" % save_path)
        
    def restore(self, task):
        # check if there is any saved model for that particular task
        if os.path.isfile("../saved-models/task-%d.ckpt.meta"%task):
            self.saver.restore(self.sess, "../saved-models/task-%d.ckpt"%task)
            print("Model restored")
        else:
            print("Saved model for given task does not exist")
        
    def test(self):
        # load the weights from the model that generated the best validation loss
        self.restore(self.task)
        start_time = time.time()
        num_chapters = self.data_and_metadata[self.datatype_id_dict['test']][5]
        total_loss, total_num_correct = self.perform_epoch(num_chapters, 'test')
        print("%d mins, Avg loss: %.2f, Num correct: %d, Accuracy: %.2f%%" % ((time.time() - start_time)/60,
                                                                          total_loss/num_chapters,
                                                                          total_num_correct,
                                                                          total_num_correct*100.0/num_chapters))
        
    def create_tensorflow_graph(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.num_sentences_in_longest_input, self.num_words_in_longest_input_sentence, self.embedding_dim])
        self.questions = tf.placeholder(tf.float32, [None, None, self.embedding_dim])
        self.answers = tf.placeholder(tf.int32, [None])
        self.input_lengths = tf.placeholder(tf.int32, [None])

        ## Question module
        with tf.variable_scope('question_module'):
            question_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_layer_size)
            _, self.question_vector = tf.nn.dynamic_rnn(question_gru_cell,
                                                  self.questions,
                                                  dtype=tf.float32)

        ## Input module
        with tf.variable_scope('input_module'):

            positionally_encoded_inputs = tf.reduce_sum(self.inputs*self.position_encoding, 2)

            input_forward_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_layer_size)
            input_backward_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_layer_size)
            input_module_output, _ = tf.nn.bidirectional_dynamic_rnn(input_forward_gru_cell,
                                                                    input_backward_gru_cell,
                                                                    positionally_encoded_inputs,
                                                                    sequence_length = self.input_lengths,
                                                                    dtype = tf.float32)
            input_fact_vectors = tf.add(input_module_output[0], input_module_output[1])
            input_fact_vectors = tf.nn.dropout(input_fact_vectors, self.dropout_probability)

        ## Episodic Memory module
        with tf.variable_scope('episodic_memory_module'):
            weight = tf.get_variable("weight", [3*self.hidden_layer_size, 80],
                                            initializer=tf.random_normal_initializer())
            bias = tf.get_variable("bias", [1, self.hidden_layer_size],
                                            initializer=tf.random_normal_initializer())
            self.previous_memory = self.question_vector
            for step in range(self.num_steps):
                attentions = []
                for fact_index, fact_vector in enumerate(tf.unstack(input_fact_vectors, axis = 1)):
                    reuse = bool(step) or bool(fact_index)
                    with tf.variable_scope("attention", reuse = reuse):
                        z = tf.concat([tf.multiply(fact_vector, self.question_vector), 
                                       tf.multiply(fact_vector, self.previous_memory),
                                       tf.abs(tf.subtract(fact_vector, self.question_vector)),
                                       tf.abs(tf.subtract(fact_vector, self.previous_memory))], 1)
                        attention = tf.contrib.layers.fully_connected(z,
                                                                    self.embedding_dim,
                                                                    activation_fn=tf.nn.tanh,
                                                                    reuse=reuse, scope="fc1")
                        attention = tf.contrib.layers.fully_connected(attention,
                                                                    1,
                                                                    activation_fn=None,
                                                                    reuse=reuse, scope="fc2")
                        attentions.append(tf.squeeze(attention))
                attentions = tf.expand_dims(tf.nn.softmax(tf.transpose(tf.stack(attentions))), axis=-1)
                reuse = True if step > 0 else False
                # soft attention
                self.context_vector = tf.reduce_sum(tf.multiply(input_fact_vectors, attentions), axis = 1)
                self.previous_memory = tf.nn.relu(tf.matmul(tf.concat([self.previous_memory, self.context_vector, self.question_vector], axis = 1), 
                                                            weight) + bias)
                        
            self.previous_memory = tf.nn.dropout(self.previous_memory, self.dropout_probability)

        ## Answer module
        with tf.variable_scope('answer_module') as scope:
            logits = tf.contrib.layers.fully_connected(inputs = tf.concat([self.previous_memory, self.question_vector], axis = 1),
                                                      num_outputs = self.vocab_size,
                                                      activation_fn = None)

            ## Loss and metrics
            self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.answers))

            # add l2 regularization for all variables except biases
            for v in tf.trainable_variables():
                if not 'bias' in v.name.lower():
                    self.loss += self.l2_regularization_lambda * tf.nn.l2_loss(v)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), 'int32')
            self.num_correct = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.answers), tf.int32))
            
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
