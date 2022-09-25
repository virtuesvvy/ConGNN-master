import os
import cv2
import random
train_txt = open(r'J:\geo_affective\cross_validation\txt\face\site_train4_16.txt', 'w')
# train_txt = open(r'J:\geo_affective\preprocess\new_site\txt\new_face\site_train_val_addneg0126.txt', 'w')
# # val_txt = open(r'E:\codes\preprocess\feature extraction\txt\face\site_val_nolstm.txt', 'w')
test_txt = open(r'J:\geo_affective\cross_validation\txt\face\site_test4_16.txt', 'w')
train_path = r'J:\geo_affective\cross_validation\patches\face\train4'
# train_path = r'J:\geo_affective\preprocess\new_site\face\train_val'
# # val_path = r'G:\geo_affective\preprocess\site\body\val'
test_path = r'J:\geo_affective\cross_validation\patches\face\test4'
# label_list = {'Negative': 0, 'Neutral': 1, 'Positive': 2, }
label_list = {'Negative': 0, 'Neutral': 1, 'Positive': 2, }
train_emotion_list = os.listdir(train_path)
face_area = []
for each_emotion in train_emotion_list:
    image_list = os.listdir(train_path + os.sep + each_emotion)
    for each_img in image_list:
        print(each_img)
        face_list = os.listdir(train_path + os.sep + each_emotion + os.sep + each_img)
        face_path = ['' for index in range(len(face_list))]
        for index in range(len(face_list)):
            face_path[index] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[index]
        if len(face_list) <= 1:
            single_face_path = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[0]
            for x in range(16):
                train_txt.write(single_face_path)
                train_txt.write(' ')
        else:
            for j in range(len(face_list) - 1):
                for i in range(len(face_list) - 1 - j):
                    if i + 1 >= len(face_list):
                        break
                    # face_path[i] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[i]
                    # face_path[i + 1] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[i + 1]
                    face = cv2.imread(face_path[i])
                    face1 = cv2.imread(face_path[i + 1])
                    if face.shape[0] * face.shape[1] < face1.shape[0] * face1.shape[1]:
                        temp_path = face_path[i]
                        face_path[i] = face_path[i + 1]
                        face_path[i + 1] = temp_path
            count = 0
            for k in range(16):
                if count == 16:
                    break
                for i in range(len(face_list)):
                    train_txt.write(face_path[i])
                    train_txt.write(' ')
                    count+=1
                    if count == 16:
                        break

        train_txt.write(str(label_list[each_emotion]))
        train_txt.write('\n')



# val_emotion_list = os.listdir(val_path)
# face_area = []
# for each_emotion in train_emotion_list:
#     image_list = os.listdir(val_path + os.sep + each_emotion)
#     for each_img in image_list:
#         face_list = os.listdir(val_path + os.sep + each_emotion + os.sep + each_img)
#         face_path = ['' for index in range(len(face_list))]
#         for index in range(len(face_list)):
#             face_path[index] = val_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[index]
#         if len(face_list) <= 1:
#             single_face_path = val_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[0]
#             for x in range(19):
#                 val_txt.write(single_face_path)
#                 val_txt.write(' ')
#         else:
#             for j in range(len(face_list) - 1):
#                 for i in range(len(face_list) - 1 - j):
#                     if i + 1 >= len(face_list):
#                         break
#                     # face_path[i] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[i]
#                     # face_path[i + 1] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[i + 1]
#                     face = cv2.imread(face_path[i])
#                     face1 = cv2.imread(face_path[i + 1])
#                     if face.shape[0] * face.shape[1] < face1.shape[0] * face1.shape[1]:
#                         temp_path = face_path[i]
#                         face_path[i] = face_path[i + 1]
#                         face_path[i + 1] = temp_path
#             count = 0
#             for k in range(19):
#                 if count == 19:
#                     break
#                 for i in range(len(face_list)):
#                     val_txt.write(face_path[i])
#                     val_txt.write(' ')
#                     count+=1
#                     if count == 19:
#                         break
#
#         val_txt.write(str(label_list[each_emotion]))
#         val_txt.write('\n')



#
test_emotion_list = os.listdir(test_path)
for each_emotion in test_emotion_list:
    image_list = os.listdir(test_path + os.sep + each_emotion)
    for each_img in image_list:
        print(each_img)
        face_list = os.listdir(test_path + os.sep + each_emotion + os.sep + each_img)
        face_path = ['' for index in range(len(face_list))]
        for index in range(len(face_list)):
            face_path[index] = test_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[index]

        if len(face_list) <= 1:
            single_face_path = test_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[0]
            for x in range(16):
                test_txt.write(single_face_path)
                test_txt.write(' ')
        else:
            for j in range(len(face_list) - 1):
                for i in range(len(face_list) - 1 - j):
                    if i + 1 >= len(face_list):
                        break
                    # face_path[i] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[i]
                    # face_path[i + 1] = train_path + os.sep + each_emotion + os.sep + each_img + os.sep + face_list[i + 1]
                    face = cv2.imread(face_path[i])
                    face1 = cv2.imread(face_path[i + 1])
                    if face.shape[0] * face.shape[1] < face1.shape[0] * face1.shape[1]:
                        temp_path = face_path[i]
                        face_path[i] = face_path[i + 1]
                        face_path[i + 1] = temp_path
            count = 0
            for k in range(16):
                if count == 16:
                    break
                for i in range(len(face_list)):
                    test_txt.write(face_path[i])
                    test_txt.write(' ')
                    count+=1
                    if count == 16:
                        break

        test_txt.write(str(label_list[each_emotion]))
        test_txt.write('\n')
