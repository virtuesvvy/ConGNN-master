"""
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1
import random

def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

# class context_encoder(feature):
#     num_kernels = [3, 32, 64, 128, 256, 256]
#     kernel_size=3, bn=True, max_pool=True, maxpool_kernel_size=2
#     padding = (kernel_size - 1) // 2
#     n = len(num_kernels) - 1
#     for i in range(n):
#         feature = tf.nn.Conv2d(feature, num_kernels[i], num_kernels[i+1], kernel_size, padding=padding)
#         if self.bn is not None:
#             feature = tf.nn.BatchNormalization(num_kernels[i+1])
#         feature = F.relu(feature)
#         if self.max_pool is not None and i < n - 1:  # check if i < n
#             feature = self.max_pool(feature)
#
#     def forward(self, x):
#         n = len(self.convs)
#         for i in range(n):
#             x = self.convs[i](x)
#             if self.bn is not None:
#                 x = self.bn[i](x)
#             x = F.relu(x)
#             if self.max_pool is not None and i < n-1:  # check if i < n
#                 x = self.max_pool(x)
#         return x



class GNN:
    def __init__(self):
        print("Start GNN Right Away!")

    def build(
            self,
            features1,
            features2,
            features3,
            face_feature_size,
            object_feature_size,
            scene_feature_size,
            hidden_size,
            num_classes,
            num_steps,
            num_face_nodes,
            num_object_nodes,
            num_scene_nodes,
            edge_features_length,
            use_bias,
            keep_prob=0.5,
            layer_num=1
    ):

        # Add an extract fully connected layer to shrink the size of features
        self.face_weights = tf.Variable(glorot_init([face_feature_size, hidden_size]), name='face_weights')
        self.face_biases = tf.Variable(np.zeros([hidden_size]).astype(np.float32), name='face_biases')
        self.object_weights = tf.Variable(glorot_init([object_feature_size, hidden_size]),
                                             name='object_weights')
        self.object_biases = tf.Variable(np.zeros([hidden_size]).astype(np.float32), name='object_biases')
        self.scene_weights = tf.Variable(glorot_init([scene_feature_size, hidden_size]), name='scene_weights')
        self.scene_biases = tf.Variable(np.zeros([hidden_size]).astype(np.float32), name='scene_biases')


        self.face_features = tf.nn.relu(tf.nn.bias_add(tf.matmul(features1, self.face_weights), self.face_biases))
        self.object_features = tf.nn.relu(tf.nn.bias_add(tf.matmul(features2, self.object_weights), self.object_biases))
        self.scene_features = tf.nn.relu(tf.nn.bias_add(tf.matmul(features3, self.scene_weights), self.scene_biases))






        # define GRU
        with tf.variable_scope("lstm_scope"):
            self.cell = tf.contrib.rnn.GRUCell(hidden_size)
            self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=keep_prob)


            # self.mlstm_cell = tf.contrib.rnn.MultiRNNCell([self.cell for _ in range(layer_num)])

        # define edge weights, edge bias, and mask used to take average of edge features.
        self.face_edge_weights = tf.Variable(glorot_init([hidden_size, edge_features_length]), name='face_edge_weights')
        self.face_edge_biases = tf.Variable(np.zeros([edge_features_length]).astype(np.float32),
                                            name='face_edge_biases')
        self.object_edge_weights = tf.Variable(glorot_init([hidden_size, edge_features_length]),
                                                  name='object_edge_weights')
        self.object_edge_biases = tf.Variable(np.zeros([edge_features_length]).astype(np.float32),
                                                 name='object_edge_biases')
        self.scene_edge_weights = tf.Variable(glorot_init([hidden_size, edge_features_length]),
                                              name='scene_edge_weights')
        self.scene_edge_biases = tf.Variable(np.zeros([edge_features_length]).astype(np.float32),
                                             name='scene_edge_biases')



        with tf.variable_scope("lstm_scope") as scope:
            mask = tf.ones(
                [num_face_nodes + num_object_nodes + 1, num_face_nodes + num_object_nodes + 1]
            ) - tf.linalg.tensor_diag(tf.ones([num_face_nodes + num_object_nodes + 1]))
            # mask_face = tf.ones(
            #     [num_face_nodes + num_object_nodes + 1, num_face_nodes + num_object_nodes + 1]
            # ) - tf.linalg.tensor_diag(tf.ones([num_face_nodes + num_object_nodes + 1]))
            # mask_context = tf.ones(
            #     [num_face_nodes + num_object_nodes + 1, num_face_nodes + num_object_nodes + 1]
            # ) - tf.linalg.tensor_diag(tf.ones([num_face_nodes + num_object_nodes + 1]))
            mask_face = tf.ones([num_face_nodes, num_face_nodes]) - tf.linalg.tensor_diag(tf.ones([num_face_nodes]))

            mask_context = tf.ones([num_object_nodes + 1, num_object_nodes + 1]
            ) - tf.linalg.tensor_diag(tf.ones([num_object_nodes + 1]))

            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                else:
                    self.state = tf.cond(
                        tf.equal(num_face_nodes, 0),
                        lambda: tf.concat([self.object_features, self.scene_features], axis=0),
                        lambda: tf.concat([self.face_features, self.object_features, self.scene_features], axis=0)
                    )

                    self.state_context = tf.cond(
                        tf.equal(num_object_nodes, 0),
                        lambda: tf.concat([self.scene_features], axis=0),
                        lambda: tf.concat([self.object_features, self.scene_features], axis=0)
                    )

                    self.state_face = self.face_features


                m_face = tf.matmul(self.state[:num_face_nodes],
                                   tf.nn.dropout(self.face_edge_weights, keep_prob=keep_prob))
                m_object = tf.matmul(
                    self.state[num_face_nodes:num_face_nodes + num_object_nodes],
                    tf.nn.dropout(self.object_edge_weights, keep_prob=keep_prob)
                )
                m_scene = tf.matmul(
                    self.state[num_face_nodes + num_object_nodes:num_face_nodes + num_object_nodes + num_scene_nodes],
                    tf.nn.dropout(self.scene_edge_weights,keep_prob=keep_prob)
                )


                if use_bias is not None:
                    m_face = tf.nn.bias_add(m_face, self.face_edge_biases)
                    m_object = tf.nn.bias_add(m_object, self.object_edge_biases)
                    m_scene = tf.nn.bias_add(m_scene, self.scene_edge_biases)

                    # tf.nn.bias_add()
                # f_i =random.randint(1,16)
                # rand_m_face = tf.slice(m_face,[f_i,0],[1,-1])
                # broad_face = tf.concat([m_face, m_face, rand_m_face], axis=0)
                # acts_face = tf.multiply(tf.matmul(mask_face, broad_face),
                #                    1 / (tf.cast(num_face_nodes + num_object_nodes + 1, tf.float32) - 1))
                acts_face = tf.multiply(tf.matmul(mask_face, m_face), 1 / (tf.cast(num_face_nodes, tf.float32) - 1))
                self.face_output, self.state_face = self.cell(acts_face, self.state_face)

                m_context =  tf.concat([m_object, m_scene], axis=0)
                # broad_context = tf.concat([m_object, m_context], axis=0)
                # acts_context = tf.multiply(tf.matmul(mask_context, broad_context),
                #                    1 / (tf.cast(num_face_nodes + num_object_nodes + 1, tf.float32) - 1))
                acts_context = tf.multiply(tf.matmul(mask_context, m_context), 1 / (tf.cast(num_object_nodes + 1, tf.float32) - 1))
                self.context_output, self.state_context = self.cell(acts_context, self.state_context)

                m_wholegraph = tf.concat([m_face, m_object, m_scene], axis=0)
                acts = tf.multiply(tf.matmul(mask, m_wholegraph),
                                   1 / (tf.cast(num_face_nodes + num_object_nodes + 1, tf.float32) - 1))
                self.rnnoutput, self.state = self.cell(acts, self.state)
                print(self.rnnoutput)
                print(3)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [hidden_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        self.logits_graph = tf.matmul(self.rnnoutput, W) + b
        self.probs_graph = tf.nn.softmax(self.logits_graph)
        self.data_dict = None
        print(self.logits_graph)

        # face stream
        with tf.variable_scope('softmax'):
            W_f = tf.get_variable('W_f', [hidden_size, num_classes])
            b_f = tf.get_variable('b_f', [num_classes], initializer=tf.constant_initializer(0.0))
        self.logits_face = tf.matmul(self.face_output, W_f) + b_f
        self.probs_face = tf.nn.softmax(self.logits_face)
        self.data_dict = None
        print(self.logits_face)

        # context stream
        with tf.variable_scope('softmax'):
            W_c = tf.get_variable('W_c', [hidden_size, num_classes])
            b_c = tf.get_variable('b_c', [num_classes], initializer=tf.constant_initializer(0.0))
        self.logits_context = tf.matmul(self.context_output, W_c) + b_c
        self.probs_context = tf.nn.softmax(self.logits_context)
        self.data_dict = None
        print(self.logits_context)



        print("build model finished")
