import os
import random

train_txt = open(r'J:\geo_affective\cross_validation\txt\object\site_train1_16.txt', 'w')
# train_txt = open(r'J:\geo_affective\preprocess\new_site\txt\new_object\site_neg_new_top16.txt', 'w')
# val_txt = open(r'E:\codes\preprocess\feature extraction\new_site\txt\new_object\site_valsz_top16.txt', 'w')
test_txt = open(r'J:\geo_affective\cross_validation\txt\object\site_test1_16.txt', 'w')
train_path = r'J:\geo_affective\cross_validation\patches\object\train1'
# train_path = r'J:\geo_affective\preprocess\new_site\object\train_val'
# val_path = r'G:\geo_affective\preprocess\new_site\object\val'
test_path = r'J:\geo_affective\cross_validation\patches\object\test1'

label_list = {'Negative': 0, 'Neutral': 1, 'Positive': 2, }
train_emotion_list = os.listdir(train_path)
for each_emotion in train_emotion_list:
    image_list = os.listdir(train_path + os.sep + each_emotion)
    for each_img in image_list:
        print(each_img)
        face_list = os.listdir(train_path + os.sep + each_emotion + os.sep + each_img)
        face_list.sort(key=lambda x: int(x[:-4]))
        count = 0
        for i, each_face in enumerate(face_list):
            face_path = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + each_face
            for k in range(16):
                if count == 16:
                    break
                for each_f in face_list:
                    f_path = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + each_f
                    train_txt.write(f_path)
                    train_txt.write(' ')
                    count += 1
                    if count == 16:
                        break
        train_txt.write(str(label_list[each_emotion]))
        train_txt.write('\n')

# val_emotion_list = os.listdir(val_path)
# for each_emotion in val_emotion_list:
#     image_list = os.listdir(val_path + os.sep + each_emotion)
#     for each_img in image_list:
#         face_list = os.listdir(val_path + os.sep + each_emotion + os.sep + each_img)
#         face_list.sort(key=lambda x: int(x[:-4]))
#         count = 0
#         for i, each_face in enumerate(face_list):
#             face_path = val_path + os.sep + each_emotion + os.sep + each_img + os.sep + each_face
#             for k in range(16):
#                 if count == 16:
#                     break
#                 for each_f in face_list:
#                     f_path = val_path + os.sep + each_emotion + os.sep + each_img + os.sep + each_f
#                     val_txt.write(f_path)
#                     val_txt.write(' ')
#                     count += 1
#                     if count == 16:
#                         break
#         val_txt.write(str(label_list[each_emotion]))
#         val_txt.write('\n')
#
test_emotion_list = os.listdir(test_path)
for each_emotion in test_emotion_list:
    image_list = os.listdir(test_path + os.sep + each_emotion)
    for each_img in image_list:
        face_list = os.listdir(test_path + os.sep + each_emotion + os.sep + each_img)
        face_list.sort(key=lambda x: int(x[:-4]))
        count = 0
        for i, each_face in enumerate(face_list):
            face_path = test_path + os.sep + each_emotion + os.sep + each_img + os.sep + each_face
            for k in range(16):
                if count == 16:
                    break
                for each_f in face_list:
                    f_path = test_path + os.sep + each_emotion + os.sep + each_img + os.sep + each_f
                    test_txt.write(f_path)
                    test_txt.write(' ')
                    count += 1
                    if count == 16:
                        break
        test_txt.write(str(label_list[each_emotion]))
        test_txt.write('\n')
