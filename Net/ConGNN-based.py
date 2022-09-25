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
import tensorflow as tf
import numpy as np
import cv2
# cv2.setNumThreads(2)
from CM_GNN_GC import GNN
import os
import tensorflow.contrib.slim as slim
import sys
from random import shuffle
import argparse
from sklearn.metrics import accuracy_score
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_data(image_path, face_data_path, object_data_path, scene_data_path, train_or_test):
    scene_list = []
    face_list = []
    object_list = []
    # skeleton_list = []
    labels = []
    if train_or_test == 'train':
        partitions = ['train']
    else:
        partitions = ['val']
    classes = ['Negative', 'Neutral', 'Positive']

    for partition in partitions:
        for i, subclass in enumerate(classes):
            image_list = os.listdir(os.path.join(image_path, partition, subclass))
            for single_image in image_list:
                image_name = single_image.split('.')[0]
                if os.path.exists(os.path.join(face_data_path, partition, subclass, image_name)):
                    face_list.append(os.path.join(face_data_path, partition, subclass, image_name))
                else:
                    face_list.append('')
                object_list.append(os.path.join(object_data_path, partition, subclass, image_name))
                scene_list.append(os.path.join(scene_data_path, partition, subclass, image_name))
                # skeleton_list.append(os.path.join(skeleton_data_path,partition, subclass, image_name))
                labels.append(i)
    assert len(object_list) == len(labels)
    assert len(face_list) == len(labels)
    assert len(scene_list) == len(labels)
    # assert len(skeleton_list) == len(labels)
    return  face_list, object_list, scene_list, labels


def shuffle_list(data_list1, data_list2, data_list3,  labels):
    c = list(zip(data_list1, data_list2, data_list3, labels))
    shuffle(c)
    data_list1, data_list2, data_list3, labels = zip(*c)
    return data_list1, data_list2, data_list3, labels


def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))

    image = (image - mean) / np.sqrt(var)

    return image


# def get_loss_PCCE(logits, labels, lambda_0):
#
#     # out = -tf.reduce_sum(logits * tf.log(labels) + (logits - 1) * tf.log(1 - labels))
#     out = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#     batch_size_list = labels.get_shape().as_list()
#     batch_size = batch_size_list[0]
#     weight = [1.] * batch_size
#     probs = tf.nn.softmax(logits)
#     _ = tf.nn.top_k(probs, k=1)
#     y_pred = _[1]
#     y_pred = tf.squeeze(y_pred)
#
#     for i in range(batch_size):
#         if (y_pred[i]==0 and labels[i]!=0) or (
#                 y_pred[i]==2 and labels[i]!=2) or (y_pred[i]==1 and labels[i]!=1):
#             weight[i] += lambda_0
#     weight_tensor = tf.cast(tf.convert_to_tensor(np.array(weight)), dtype=tf.float32)
#     # weight_tensor = tf.reshape(weight_tensor, [batch_size,1])
#     # out = tf.reshape(out, [1,batch_size])
#     out = tf.multiply(out, weight_tensor)
#     out = tf.reduce_mean(out)
#
#     return out

def get_loss(logits_g, logits_f, logits_c, labels_g, labels_f, labels_c, probs_g, probs_f, probs_c):

    loss_graph = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_g, labels=labels_g)
    loss_face = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_f, labels=labels_f)
    loss_context = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_c, labels=labels_c)

    loss_f = tf.reduce_mean(loss_face)
    loss_c = tf.reduce_mean(loss_context)
    loss_g = tf.reduce_mean(loss_graph)

    probs_c_m = tf.reduce_mean(probs_c, 0)
    # probs_c_m = np.mean(probs_c, axis=0)
    pred_context = np.argmax(probs_c_m, 0)
    # print('pred_c', pred_context)

    probs_f_m = tf.reduce_mean(probs_f,0)
    pred_face = np.argmax(probs_f_m, 0)

    if pred_context == pred_face:
        lambda_0 = 0
    else:
        lambda_0 = 1

    loss_CCE = (1 + 0.2* lambda_0) * (loss_f + loss_g + loss_c)

    return loss_f, loss_c, loss_CCE,loss_g, probs_c_m


def main(args):
    tf.reset_default_graph()
    train_or_test = args.train_or_test
    num_epoches = 60
    face_feature_len = 256
    object_feature_len = 1024
    scene_feature_len = 1024
    skeleton_feature_len = 1024
    num_class = 3
    max_object_nodes = 16
    if train_or_test == 'train':
        max_face_nodes = 16
    else:
        max_face_nodes = 48

    X1 = tf.placeholder("float", [16, face_feature_len])
    X2 = tf.placeholder("float", [16, object_feature_len])
    X3 = tf.placeholder("float", [1, scene_feature_len])
    X4 = tf.placeholder("float", [1, skeleton_feature_len])
    Y = tf.placeholder("float", [33])
    Y_face = tf.placeholder("float", [16])
    Y_context = tf.placeholder("float", [17])
    dropout_flag = tf.placeholder_with_default(0.5, shape=())
    # node_len_face = tf.placeholder("int32", shape=())
    # node_len_object = tf.placeholder("int32", shape=())
    node_len_face = 16
    node_len_object = 16
    node_len_scene = 1
    net = GNN()

    with tf.name_scope("my_model"):
        net.build(
            features1=X1,
            features2=X2,
            features3=X3,
            # features4= X4,
            face_feature_size=face_feature_len,
            object_feature_size=object_feature_len,
            scene_feature_size=scene_feature_len,
            # skeleton_feature_size = skeleton_feature_len,
            hidden_size=128,
            edge_features_length=128,
            layer_num=1,
            num_face_nodes=node_len_face,
            num_object_nodes=node_len_object,
            num_scene_nodes = node_len_scene,
            use_bias=False,
            keep_prob=dropout_flag,
            num_classes=num_class,
            num_steps=4,
        )

    print('\ntrainable variables')
    for v in tf.trainable_variables():
        print(v.name, v.get_shape().as_list())

    # learning_rate = 0.0002
    # learning_rate_base = 0.001
    # learning_decay_steps = 10000
    # learning_decay_rate = 0.95
    #
    # learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, learning_decay_steps,
    #                                            learning_decay_rate, staircase=True)
    # learning_rate_base = 0.0003
    # learning_decay_steps = 10*46837
    # learning_decay_rate = 0.9
    #              # global_step = 18221
    # global_step = tf.Variable(0, trainable=False)
    #
    # learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, learning_decay_steps,
    #                                                                             learning_decay_rate, staircase=True)
    #


    warmup_step = 10*8012
    learning_rate_base = 0.0002
    global_step = tf.Variable(0, trainable=False)
    c = (global_step/8012 + 1) * 8012
    learning_rate_step = 8012
    learning_rate_decay = 0.95
    staircase=False
    epoch_index = 0
    #46837
    linear_increase = learning_rate_base * tf.cast(c, tf.float32) / tf.cast(warmup_step, tf.float32)
    linear_increase = learning_rate_base * (tf.cast(global_step, dtype=tf.float32) + 8012) / warmup_step

    exponential_decay = tf.train.exponential_decay(learning_rate_base,
                                                   c - warmup_step,
                                                   learning_rate_step,
                                                   learning_rate_decay,
                                                   staircase=staircase)
    learning_rate = tf.cond(global_step <= warmup_step,
                            lambda: linear_increase,
                            lambda: exponential_decay)


    corr = tf.equal(tf.argmax(net.probs_graph, 1), tf.to_int64(Y))
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))



    # loss = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.logits, labels=tf.to_int64(Y)))
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.logits, labels=tf.to_int64(Y))
    # with tf.name_scope('loss'):
    loss_f, loss_c, loss_CCE,loss_g, probs_c_m = get_loss(logits_g=net.logits_graph, logits_f=net.logits_face, logits_c=net.logits_context,
                    labels_g=tf.to_int64(Y),labels_f=tf.to_int64(Y_face),labels_c=tf.to_int64(Y_context),
                    probs_g=net.probs_graph, probs_f=net.probs_face, probs_c=net.probs_context)
    # loss = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.logits, labels=tf.to_int64(Y)))
    # tf.summary.scalar('Loss', loss)
    #
    # train_op_f = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_f,global_step=global_step)
    # train_op_c = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_c,global_step=global_step)
    # train_CEE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_CCE,global_step=global_step)
    # train_op_g = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_g,global_step=global_step)
    # train_op = tf.group(train_CEE,train_op_f,train_op_c,train_op_g)

    train_op_f = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_f,global_step=global_step)
    train_op_c = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_c,global_step=global_step)
    train_CEE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_CCE,global_step=global_step)
    train_op_g = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_g,global_step=global_step)
    train_op = tf.group(train_CEE,train_op_f,train_op_c,train_op_g)

    # train_op = tf.group(train_op_f, train_op_c,train_op_cee, train_op_g)


    # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_CCE)

    # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    # init = tf.global_variables_initializer()
    # saver_restore = tf.train.Saver(var_list=variables_to_restore)

    # saver = tf.train.Saver(max_to_keep=50)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=60)

    label_list = {0: 'Negative', 1: 'Neutral', 2: 'Positive',
                  'Negative': 0, 'Neutral': 1, 'Positive': 2}
    # txt_path = open(r'E:\pky\GANN_data\label2predict_sitesplus.txt', 'w')
    # variable_restore_op = slim.assign_from_checkpoint_fn('./model_weights/resnet_v1_50.ckpt',
    #                                                      slim.get_trainable_variables(),
    #                                                      ignore_missing_vars=True)  # 一定要设置为True
    # cpu_num = int(os.environ.get('CPU_NUM', 12))
    # config = tf.ConfigProto(device_count={"CPU": cpu_num},inter_op_parallelism_threads=cpu_num,intra_op_parallelism_threads=cpu_num,log_device_placement=True)
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        epoch_index += 1
        print("start")
        # sess.run(init)
        org_path = args.org_path
        face_path = args.face_feature_path
        object_path = args.object_feature_path
        scene_path = args.scene_feature_path
        # skeleton_path = args.skeleton_feature_path



        # variable_restore_op = slim.assign_from_checkpoint_fn('./model_weights/resnet_v1_50.ckpt',
        #                                                      slim.get_trainable_variables(),
        #                                                      ignore_missing_vars=True)  # 一定要设置为True
        # if
        saver.restore(sess, os.path.join('E:/codes/geo_affective/GCNN_Master/fso1/Gro_cml_fso_0111_4losswarm_lr2', "model_epoch" + str(11) + ".ckpt"))
        sess.run(tf.global_variables_initializer())


        (face_whole_list,object_whole_list,scene_whole_list, label_whole_list ) = \
            read_data(org_path, face_path, object_path, scene_path, train_or_test)

        num_samples = len(face_whole_list)
        # print(num_samples)

        # if train_or_test in {'test', 'val'}:
        #    np.save(os.path.join(args.model_path, train_or_test + 'labels.npy', label_list))

        for epoch in range(num_epoches):
            count = 0
            # print(sess.run(c))
            # print(sess.run(learning_rate))
            # tf.cast(learning_rate)
            # print(sess.run(linear_increase))
            # print(sess.run(exponential_decay))
            # print(sess.run(global_step))

            if train_or_test == 'train':
                (
                    face_whole_list,
                    object_whole_list,
                    scene_whole_list,
                    label_whole_list,
                ) = shuffle_list(
                    face_whole_list,
                    object_whole_list,
                    scene_whole_list,
                    label_whole_list
                )
                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)
            else:
                saver.restore(sess, os.path.join(args.model_path, "model_epoch" + str(epoch + 1) + ".ckpt"))
                probs_whole = []

            while count < num_samples:

                # face
                batch_face_list = face_whole_list[count]
                face_list = os.listdir(batch_face_list)
                if len(face_list) == 0:
                    # face_nodes_len = 0
                    batch_face_x = np.zeros((1, face_feature_len))


                else:
                    # face_list = os.listdir(batch_face_list)
                    # face_nodes_len = len(face_list)
                    batch_face_x = []
                    # if train_or_test == 'train' and face_nodes_len> max_face_nodes:
                    if train_or_test == 'train':
                        shuffle(face_list)

                    # face_nodes_len = face_nodes_len if face_nodes_len < max_face_nodes else max_face_nodes
                    face_nodes_len = 16
                    face_len = len(face_list)
                    for j in range(face_nodes_len):
                        batch_face_x.append(
                            np.reshape(
                                np.load(os.path.join(batch_face_list, face_list[j%face_len]), allow_pickle=True),
                                [face_feature_len, ]
                            )
                        )

                # context
                batch_object_list = object_whole_list[count]
                object_list = os.listdir(batch_object_list)
                # assert len(attention_list) == max_attention_nodes
                object_nodes_len = len(object_list)
                batch_object_x = []
                # print(batch_attention_list)
                for j in range(16):
                    batch_object_x.append(
                        np.reshape(
                            np.load(os.path.join(batch_object_list, object_list[j%object_nodes_len]),allow_pickle=True),
                            [object_feature_len,]
                        )
                    )

                # batch_object_list = object_whole_list[count]
                # batch_object_x = np.reshape(
                #     np.load(batch_object_list + os.sep + batch_object_list.split('\\')[-1] + '.npy'),
                #     [1, object_feature_len])


                batch_scene_list = scene_whole_list[count]
                batch_scene_x = np.reshape(
                    np.load(batch_scene_list + os.sep + batch_scene_list.split('\\')[-1] + '.npy'),
                    [1, scene_feature_len])

                # batch_skeleton_list = skeleton_whole_list[count]
                # batch_skeleton_x = np.reshape(
                #     np.load(batch_skeleton_list + os.sep + batch_skeleton_list.split('\\')[-1] + '.npy'),
                #     [1, skeleton_feature_len])

                batch_y = np.repeat(label_whole_list[count], face_nodes_len + 17, axis=0)
                batch_y_f = np.repeat(label_whole_list[count], face_nodes_len, axis=0)
                batch_y_c = np.repeat(label_whole_list[count], 17, axis=0)
                # print(batch_scene_list)

                if train_or_test == 'train':
                    # probs_out = sess.run(net.probs,
                    #                      feed_dict={
                    #                          X1: batch_face_x,
                    #                          X2: batch_object_x,
                    #                          X3: batch_scene_x,
                    #                          X4: batch_fa_x,
                    #                          Y: batch_y,
                    #                          dropout_flag: 1.0,
                    #                          # node_len_face: face_nodes_len,
                    #                          # node_len_object: object_nodes_len
                    #                      }
                    #                      )
                    sess.run(train_op,
                             feed_dict={
                                 X1: batch_face_x,
                                 X2: batch_object_x,
                                 X3: batch_scene_x,
                                 # X4: None,
                                 Y: batch_y,
                                 Y_face: batch_y_f,
                                 Y_context: batch_y_c,
                                 dropout_flag: 0.5,
                                 # node_len_face:face_nodes_len,
                                 # node_len_object:object_nodes_len
                             }
                             )

                else:
                    probs_out = sess.run(net.probs,
                                         feed_dict={
                                             X1: batch_face_x,
                                             X2: batch_object_x,
                                             X3: batch_scene_x,
                                             X4: batch_skeleton_x,
                                             Y: batch_y,
                                             dropout_flag: 1.0,
                                             # node_len_face: face_nodes_len,
                                             # node_len_object: object_nodes_len
                                         }
                                         )
                    probs = np.mean(probs_out, axis=0)
                    probs_whole.append(probs)
                    # predict = np.argmax(probs, 0)
                    #
                    # txt_path = open(r'E:\pky\GANN_data\plus\label2predict_sites'+str(i)+'.txt', 'w')
                    # txt_path.write(
                    #     'val_' + batch_scene_list.split('\\')[-2] + '_' + batch_scene_list.split('\\')[-1] + '.jpg')
                    # txt_path.write(' ')
                    # txt_path.write(str(label_list[batch_scene_list.split('\\')[-2]]))
                    # txt_path.write(' ')
                    # txt_path.write(str(predict))
                    # txt_path.write('\n')

                count += 1

                if train_or_test == 'train' and count % (100) == 0:
                    train_accuracy = sess.run(
                        accuracy,
                        feed_dict={
                            X1: batch_face_x,
                            X2: batch_object_x,
                            X3: batch_scene_x,
                            # X4: batch_skeleton_x,
                            Y: batch_y,
                            dropout_flag: 1.0,
                            # node_len_face: face_nodes_len,
                            # node_len_object: object_nodes_len
                        }
                    )
                    #
                    # pre_c = sess.run(
                    #     probs_c_m,
                    #     feed_dict={
                    #         X1: batch_face_x,
                    #         X2: batch_object_x,
                    #         X3: batch_scene_x,
                    #         X4: batch_skeleton_x,
                    #         Y: batch_y,
                    #         dropout_flag: 1.0,
                    #         # node_len_face: face_nodes_len,
                    #         # node_len_object: object_nodes_len
                    #     }
                    # )

                    print(" Step %d, training accuracy %f" % (count, train_accuracy))

            if train_or_test == 'train':
                model_name = os.path.join(args.model_path, "model_epoch" + str(epoch + 1) + ".ckpt")
                saver.save(sess, model_name)
            else:
                assert len(probs_whole) == num_samples
                # probs_whole=probs_whole.reshape((len(probs_whole),3))
                predicts = np.argmax(probs_whole, 1)
                # np.save(os.path.join(args.model_path, "model_epoch"+str(epoch+1)+'.npy'), probs_whole)
                print("Epoch " + str(epoch + 1) + " accuracy is %f" % accuracy_score(label_whole_list, predicts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN')
    parser.add_argument('--train_or_test', type=str, default='train', choices={'train', 'test'},
                        help='Train or test flag.')
    # parser.add_argument('--org_path', type=str, default='/data/yuanyuan/vvy/data/new_site/scene',
    #                     help='path to original data.')
    # parser.add_argument('--face_feature_path', type=str,
    #                     default='/data/yuanyuan/vvy/data/new_site/feature/lstm',
    #                     help='path to the face features.')
    # parser.add_argument('--object_feature_path', type=str,
    #                     default='/data/yuanyuan/vvy/data/new_site/feature/object',
    #                     help='path to the object features.')
    # parser.add_argument('--scene_feature_path', type=str,
    #                     default='/data/yuanyuan/vvy/data/new_site/feature/scene',
    #                     help='path to the scene features.')
    # parser.add_argument('--skeleton_feature_path', type=str, default='/data/yuanyuan/vvy/data/new_site/feature/skeleton',
    #                     help='path to the skeleton features.')
    #
    # parser.add_argument('--org_path', type=str, default='F:/CM_GCN/groupEmoW_feature/GroupEmoW_rgb',
    #                     help='path to original data.')
    # parser.add_argument('--face_feature_path', type=str,
    #                     default='E:/codes/preprocess/feature extraction/features/LSTM_new_retrain',
    #                     help='path to the face features.')
    # parser.add_argument('--object_feature_path', type=str,
    #                     default='E:/codes/preprocess/feature extraction/features/GroEmoW_object1031',
    #                     help='path to the object features.')
    # parser.add_argument('--scene_feature_path', type=str,
    #                     default='E:/codes/preprocess/feature extraction/features/groupEmoW_scene106',
    #                     help='path to the scene features.')
    # parser.add_argument('--skeleton_feature_path', type=str, default='F:/CM_GCN/groupEmoW_feature/GroupEmoW/skeleton2',
    #                     help='path to the skeleton features.')

    #
    # parser.add_argument('--org_path', type=str, default='J:/geo_affective/preprocess/new_site/scene',
    #                     help='path to original data.')
    # parser.add_argument('--face_feature_path', type=str,
    #                     default='J:/geo_affective/preprocess/new_site/features/lstm_new',
    #                     help='path to the face features.')
    # parser.add_argument('--object_feature_path', type=str,
    #                     default='J:/geo_affective/preprocess/new_site/features/object_new',
    #                     help='path to the object features.')
    # parser.add_argument('--scene_feature_path', type=str,
    #                     default='J:/geo_affective/preprocess/new_site/features/scene_new',
    #                     help='path to the scene features.')
    # parser.add_argument('--skeleton_feature_path', type=str, default='J:/geo_affective/preprocess/new_site/features/skeleton',
    #                     help='path to the skeleton features.')
    #
    parser.add_argument('--org_path', type=str, default='J:/geo_affective/cross_validation/data',
                        help='path to original data.')
    parser.add_argument('--face_feature_path', type=str,
                        default='J:/geo_affective/cross_validation/features/cross3/LSTM',
                        help='path to the face features.')
    parser.add_argument('--object_feature_path', type=str,
                        default='J:/geo_affective/cross_validation/features/cross3/object',
                        help='path to the object features.')
    parser.add_argument('--scene_feature_path', type=str,
                        default='J:/geo_affective/cross_validation/features/cross3/scene',
                        help='path to the scene features.')
    parser.add_argument('--model_path', type=str, default='J:/geo_affective/preprocess/new_site/weights/cross/3/cml_0531',
                        help='path to save the generated models or to the saved models.')

    args = parser.parse_args()

    main(args)



