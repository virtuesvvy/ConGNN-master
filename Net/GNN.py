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
import shutil
from sklearn.metrics import accuracy_score
import datetime

def read_data(image_path, face_data_path, object_data_path, scene_data_path):
    img_jpg_path = []
    scene_list = []
    face_list = []
    object_list = []
    partitions = os.listdir(image_path)

    for partition in partitions:
        img_path = os.path.join(image_path, partition)
        image_list = os.listdir(img_path)
        image_list.sort(key=lambda x: int(x[:-4]))
        for single_image in image_list:
            image_name = single_image.split('.')[0]
            if os.path.exists(os.path.join(face_data_path, partition, image_name)):
                img_jpg_path.append(os.path.join(img_path, single_image))
                face_list.append(os.path.join(face_data_path, partition, image_name))
            else:
                face_list.append('')
            object_list.append(os.path.join(object_data_path, partition, image_name))
            scene_list.append(os.path.join(scene_data_path, partition, image_name))


    return  img_jpg_path, face_list, object_list, scene_list



def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))

    image = (image - mean) / np.sqrt(var)

    return image


def main(args):
    tf.reset_default_graph()
    face_feature_len = 256
    object_feature_len = 1024
    scene_feature_len = 1024
    skeleton_feature_len = 1024
    num_class = 3

    X1 = tf.placeholder("float", [16, face_feature_len])
    X2 = tf.placeholder("float", [16, object_feature_len])
    X3 = tf.placeholder("float", [1, scene_feature_len])
    X4 = tf.placeholder("float", [1, skeleton_feature_len])

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


    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=50)
    label_list = {0: 'Negative', 1: 'Neutral', 2: 'Positive',
                  'Negative': 0, 'Neutral': 1, 'Positive': 2}
    txt_path = open(r'E:/codes/geo_affective/GCNN_Master/data/label2predict/cross/cross3/cml_fso_0601.txt','w')
    # tsne_path = open(r'E:/codes/geo_affective/GCNN_Master/fso/grow_GNN_fso_0111_1loss_lr1ep4.txt', 'a+',encoding='utf-8')

    with tf.Session() as sess:
        print("start")
        # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        # sess = tf.compat.v1.Session(config=config)
        sess.run(init)
        org_path = args.org_path
        face_path = args.face_feature_path
        object_path = args.object_feature_path
        scene_path = args.scene_feature_path


        (img_path, face_whole_list,object_whole_list,scene_whole_list) = \
            read_data(org_path, face_path, object_path, scene_path)

        num_samples = len(face_whole_list)
        for i in range(100):
            i += 1
            saver.restore(sess, os.path.join(args.model_path, "model_epoch" + str(i) + ".ckpt"))

            # if train_or_test in {'test', 'val'}:
            #    np.save(os.path.join(args.model_path, train_or_test + 'labels.npy', label_list))

            count = 0
            true_pred = 0
            test_acc = 0
            while count < num_samples:
                batch_img_list = img_path[count]

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
                    # if train_or_test == 'train':
                    #     shuffle(face_list)

                    # face_nodes_len = face_nodes_len if face_nodes_len < max_face_nodes else max_face_nodes
                    face_nodes_len = 16
                    face_len = len(face_list)
                    for j in range(face_nodes_len):
                        batch_face_x.append(
                            np.reshape(
                                np.load(os.path.join(batch_face_list, face_list[j % face_len]), allow_pickle=True),
                                [face_feature_len, ]
                            )
                        )

                # context

                batch_object_list = object_whole_list[count]
                object_list = os.listdir(batch_object_list)
                # assert len(object_list) == max_object_nodes
                object_nodes_len = len(object_list)
                batch_object_x = []
                # print(batch_object_list)
                for j in range(16):
                    batch_object_x.append(
                        np.reshape(
                            np.load(os.path.join(batch_object_list, object_list[j % object_nodes_len]),
                                    allow_pickle=True),
                            [object_feature_len, ]
                        )
                    )
                #
                # batch_object_list = object_whole_list[count]
                # batch_object_x = np.reshape(
                #     np.load(batch_object_list + os.sep + batch_object_list.split('\\')[-1] + '.npy'),
                #     [1, object_feature_len])

                batch_scene_list = scene_whole_list[count]
                batch_scene_x = np.reshape(
                    np.load(batch_scene_list + os.sep + batch_scene_list.split('\\')[-1] + '.npy'),
                    [1, scene_feature_len])


                # print(batch_scene_list)
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # print(time)

                probs_out = sess.run(net.probs_graph,
                                     feed_dict={
                                         X1: batch_face_x,
                                         X2: batch_object_x,
                                         X3: batch_scene_x,
                                         # X4: None,
                                         dropout_flag: 1.0,
                                         # node_len_face: face_nodes_len,
                                         # node_len_object: object_nodes_len
                                     }
                                     )

                # probs_out.reshape([-1, 1])
                np.set_printoptions(linewidth=np.inf)
                # print("{}".format(np.array(probs_out).flatten()), file=tsne_path)
                # tsne_path.write(np.array(probs_out).flatten())
                # tsne_path.write('\n')
                probs = np.mean(probs_out, axis=0)

                predicts = np.argmax(probs, 0)
                txt_path.write('test_' + batch_scene_list.split('\\')[-2] + '_' + batch_scene_list.split('\\')[-1] + '.jpg')
                txt_path.write(str(label_list[batch_scene_list.split('\\')[-2]]))
                a = str(label_list[batch_scene_list.split('\\')[-2]])
                txt_path.write(' ')
                txt_path.write(str(predicts))
                b = str(predicts)
                txt_path.write('\n')
                print(batch_scene_list.split('\\')[-2] + batch_scene_list.split('\\')[-1])
                orgfilepath, filename = os.path.split(batch_img_list)
                if a == b:
                    true_pred += 1
                # if a != b:
                # des = os.path.join('J:/Geo_affective/preprocess/new_site/pred/gro_cml_test', str(a), str(predicts))
                # if not os.path.exists(des):
                #    os.mkdir(des)
                # shutil.copy(batch_img_list, os.path.join(des, filename))

                # path = os.path.join(org_path, batch_scene_list.split('\\')[-2],
                #                     batch_scene_list.split('\\')[-1] + '.png')
                # to_path = os.path.join(des, batch_scene_list.split('\\')[-2], label_list[predicts])
                # if not os.path.exists(to_path):
                #     os.makedirs(to_path)
                # shutil.copy(path, to_path)
                # print(batch_scene_list.split('\\')[-2] + os.sep + batch_scene_list.split('\\')[-1] + '.png' + '  is ' +
                #       label_list[predicts])
                count += 1
            test_acc = true_pred / num_samples
            txt_path.write('test_accuracy: ' + 'ep_' + str(i) + ': ' + str(test_acc))
            txt_path.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GNN')
    parser.add_argument('--train_or_test', type=str, default='test', choices={'train', 'test'},
                        help='Train or test flag.')

    # parser.add_argument('--org_path', type=str, default='F:/CM_GCN/groupEmoW_feature/GroupEmoW_rgb/test',
    #                     help='path to original data.')
    # parser.add_argument('--face_feature_path', type=str,
    #                     default='E:/codes/preprocess/feature extraction/features/LSTM_new_retrain/test',
    #                     help='path to the face features.')
    # parser.add_argument('--object_feature_path', type=str,
    #                     default='E:/codes/preprocess/feature extraction/features/GroEmoW_object1031/test',
    #                     help='path to the object features.')
    # parser.add_argument('--scene_feature_path', type=str,
    #                     default='E:/codes/preprocess/feature extraction/features/groupEmoW_scene106/test',
    #                     help='path to the scene features.')
    # parser.add_argument('--skeleton_feature_path', type=str, default='F:/CM_GCN/groupEmoW_feature/GroupEmoW/skeleton2/test',
    #                     help='path to the skeleton features.')
    #
    # parser.add_argument('--org_path', type=str, default='J:/geo_affective/preprocess/new_site/scene/test',
    #                     help='path to original data.')
    # parser.add_argument('--face_feature_path', type=str,
    #                     default='J:/geo_affective/preprocess/new_site/features/lstm_new/test',
    #                     help='path to the face features.')
    # parser.add_argument('--object_feature_path', type=str,
    #                     default='J:/geo_affective/preprocess/new_site/features/object_new/test',
    #                     help='path to the object features.')
    # parser.add_argument('--scene_feature_path', type=str,
    #                     default='J:/geo_affective/preprocess/new_site/features/scene_new/test',
    #                     help='path to the scene features.')
    # parser.add_argument('--skeleton_feature_path', type=str, default='J:/geo_affective/preprocess/new_site/features/skeleton/test',
    #                     help='path to the skeleton features.')

    parser.add_argument('--org_path', type=str, default='J:/geo_affective/cross_validation/data/test',
                        help='path to original data.')
    parser.add_argument('--face_feature_path', type=str,
                        default='J:/geo_affective/cross_validation/features/cross3/LSTM/test',
                        help='path to the face features.')
    parser.add_argument('--object_feature_path', type=str,
                        default='J:/geo_affective/cross_validation/features/cross3/object/test',
                        help='path to the object features.')
    parser.add_argument('--scene_feature_path', type=str,
                        default='J:/geo_affective/cross_validation/features/cross3/scene/test',
                        help='path to the scene features.')
    parser.add_argument('--model_path', type=str, default='J:/geo_affective/preprocess/new_site/weights/cross/3/cml_0531',
                        help='path to save the generated models or to the saved models.')

    args = parser.parse_args()

    main(args)
