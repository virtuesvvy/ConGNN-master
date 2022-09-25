import os
import random

train_txt = open(r'J:\geo_affective\cross_validation\txt\scene\site_train4_16.txt', 'w')
# train_txt = open(r'J:\geo_affective\preprocess\new_site\txt\new_scene\site_train0125newneg.txt', 'w')
test_txt = open(r'J:\geo_affective\cross_validation\txt\scene\site_test4_16.txt', 'w')
# # val_txt = open(r'F:\CM_GCN\groupEmoW_feature\GroupEmoW\txt\skeleton\site_val_skel.txt', 'w')
train_path = r'J:\geo_affective\cross_validation\data\train4'
# train_path = r'J:\geo_affective\preprocess\new_site\scene\train'
test_path = r'J:\geo_affective\cross_validation\data\test4'
# # val_path = r'D:\data\dataset\site\skeleton\val'

label_list = {'Negative': 0, 'Neutral': 1, 'Positive': 2, }
train_emotion_list = os.listdir(train_path)
for each_emotion in train_emotion_list:
    image_list = os.listdir(train_path + os.sep + each_emotion)
    for each_img in image_list:
        face_path = train_path + os.sep + each_emotion + os.sep + each_img
        train_txt.write(face_path)
        train_txt.write(' ')
        train_txt.write(str(label_list[each_emotion]))
        train_txt.write('\n')

test_emotion_list = os.listdir(test_path)
for each_emotion in test_emotion_list:
    image_list = os.listdir(test_path + os.sep + each_emotion)
    for each_img in image_list:
        face_path = test_path + os.sep + each_emotion + os.sep + each_img
        test_txt.write(face_path)
        test_txt.write(' ')
        test_txt.write(str(label_list[each_emotion]))
        test_txt.write('\n')

# val_emotion_list = os.listdir(val_path)
# for each_emotion in val_emotion_list:
#     image_list = os.listdir(val_path + os.sep + each_emotion)
#     for each_img in image_list:
#         face_path = val_path + os.sep + each_emotion + os.sep + each_img
#         val_txt.write(face_path)
#         val_txt.write(' ')
#         val_txt.write(str(label_list[each_emotion]))
#         val_txt.write('\n')
