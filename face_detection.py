import cv2
import os
import numpy as np
import csv
from pathlib import Path
from matplotlib import pyplot as plt
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

g_embedder = FaceNet()
g_lencoder = LabelEncoder()


def load_image(img_path: Path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_dataset_csv(csv_file: Path) -> list:
    data = []
    with open(csv_file, mode='r') as fp_csv:
        csv_file = csv.reader(fp_csv, delimiter=";")
        for lines in csv_file:
            p = Path(lines[0])
            if not p.exists():
                raise Exception(f"Wrong image file path {p}")
            data.append({'full_path': p, 'sub_num': lines[1]})
    
    imgs = []
    labels = []
    for field in data:
        imgs.append(load_image(Path(field['full_path'])))
        labels.append(f"{field['full_path'].parent}")
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    
    return labels, imgs


def extract_dataset_csv(csv_file: Path) -> tuple:
    data = []
    with open(csv_file, mode='r') as fp_csv:
        csv_file = csv.reader(fp_csv, delimiter=";")
        for lines in csv_file:
            p = Path(lines[0])
            if not p.exists():
                raise Exception(f"Wrong image file path {p}")
            data.append({'full_path': p, 'sub_num': lines[1]})
    
    all_labels, all_results = [], []
    for field in data:
        labels, results = extract_faces(cv2.imread(field['full_path']), field['full_path'])
        all_labels.extend(labels)
        all_results.extend(results)
    
    return all_labels, all_results


def extract_faces(img, img_path: Path) -> tuple:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 4) # 1.5 powinno być zkalibrowane w zależności od zdjęcia wejściowego

    labels = []
    cropped_imgs = []
    for i, (x, y, w, h) in enumerate(faces):
        crop = cv2.resize(img[y:y+h, x:x+w], (160, 160))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cropped_imgs.append(crop)
        labels.append(f"{img_path.parent}-{i}")
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    labels = np.asarray(labels)
    cropped_imgs = np.asarray(cropped_imgs)

    return labels, cropped_imgs


def plot_images(imgs: list, ncols = 3, fig_w = 18, fig_h = 12):
    plt.figure(figsize=(fig_w, fig_h))
    for i, img in enumerate(imgs):
        nrows = (len(imgs) // ncols) + 1
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.waitforbuttonpress()


def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = g_embedder.embeddings(face_img)
    return yhat[0]


def SVM_train(labels, embeddings):
    g_lencoder.fit(labels)
    labels = g_lencoder.transform(labels)

    # Training
    IMG_train, IMG_test, LBL_train, LBL_test = train_test_split(embeddings, labels, shuffle=True, random_state=17)
    model = SVC(kernel='linear', probability=True)
    model.fit(IMG_train, LBL_train)
    lbl_preds_train = model.predict(IMG_train)
    lbl_preds_test = model.predict(IMG_test)

    acc = accuracy_score(LBL_test, lbl_preds_test)
    print(f"Accuracy: {acc*100}%")


def compare_videos(video1: Path, video2: Path):
    if not video1.exists():
        print(f"{video_path1} doesn't exist!")
    if not video2.exists():
        print(f"{video_path2} doesn't exist!")

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    plot_data = {
        'x' : [],
        'y' : []
        }
    i = 1
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Can't receive frame from video stream")
            break

        if i % 5: # set fps
            i += 1
            continue

        _, lface1 = extract_faces(frame1, video1)
        _, lface2 = extract_faces(frame2, video2)
        if len(lface1) > 1 or len(lface2) > 1:
            print("Frame has more than one face!")

        if lface1.size == 0 or lface2.size == 0:
            print("Frame has none faces!")
            continue

        face1 = lface1[0]
        face2 = lface2[0]
        emb_face1 = get_embedding(face1)
        emb_face2 = get_embedding(face2)
        diff = g_embedder.compute_distance(emb_face1, emb_face2)
        plot_data['x'].append(i)
        plot_data['y'].append(diff)

        frame_cat = np.hstack((face1, face2)) 
        cv2.imshow("Frame comparison", frame_cat)

        if cv2.waitKey(1) == ord('q'):
            break
        i += 1
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    plt.plot(plot_data['x'], plot_data['y'])
    plt.xlabel('Frame num')
    plt.ylabel('Embeddings difference')
    plt.title('Embedding difference across video frames')
    plt.show()


if __name__ == "__main__":
    video_path1 = Path().joinpath("data", "Videos", "SpoofEmotions.mov")
    video_path2 = Path().joinpath("data", "Videos", "OriginalEmotions.mov")
    compare_videos(video_path1, video_path2)

# ========================== EXAMPLES ========================
# labels, imgs = load_dataset_csv(Path().cwd().joinpath("file_label.csv"))
# embeddings = []
# for cropped_face in imgs:
#     embeddings.append(get_embedding(cropped_face))
# embeddings = np.asarray(embeddings)
# np.savez_compressed('embeddings_faces.npz', embeddings, labels)

# npdata = np.load('embeddings_faces.npz')
# embeddings = npdata['arr_0']
# labels = npdata['arr_1']
# print(f"len: {len(embeddings)}")

# print(f"DISTANCES:")
# print(f"0 vs 5: {g_embedder.compute_distance(embeddings[0], embeddings[5])}")
# print(f"10 vs 13: {g_embedder.compute_distance(embeddings[10], embeddings[13])}")
# print(f"0 vs 13: {g_embedder.compute_distance(embeddings[0], embeddings[13])}")

# plot_images(imgs, 10, 20)
# plot_images([imgs[0], imgs[5]])
# plot_images([imgs[10], imgs[13]])
# plot_images([imgs[0], imgs[13]])