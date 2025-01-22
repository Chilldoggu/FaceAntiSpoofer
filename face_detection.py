import cv2
import os
import numpy as np
import csv
from mtcnn.mtcnn import MTCNN
from pathlib import Path
from matplotlib import pyplot as plt
from keras_facenet import FaceNet
from matplotlib.animation import FFMpegWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
plt.rcParams['animation.ffmpeg_path'] = 'E:\\PWr\\SEM5\\Biometria\\FaceAntiSpoofer\\bin\\ffmpeg.exe'

g_embedder = FaceNet()
g_mtcnn_detector = MTCNN()

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


def extract_faces(img, img_path: Path, scale_fact = 1.3, bgr2rgb = True) -> tuple:
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scale_fact, 4) # drugi parametr powinien być zkalibrowany w zależności od zdjęcia wejściowego

    faces = g_mtcnn_detector.detect_faces(img)

    labels = []
    cropped_imgs = []
    for i, face in enumerate(faces):
        (x, y, w, h) = face['box']
        crop = cv2.resize(img[y:y+h, x:x+w], (160, 160))
        if bgr2rgb:
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


def compare_videos(video1: Path, video2: Path, scale_fact = 1.3, result_path = Path()):
    if not video1.exists():
        print(f"{video1} doesn't exist!")
    if not video2.exists():
        print(f"{video2} doesn't exist!")
    
    if result_path.exists() is False:
        result_path.mkdir(parents=True, exist_ok=True)

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    plot_data = { 'x' : [], 'y' : [] }
    i = 1
    j = 0
    err = 0
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Can't receive frame from video stream")
            break

        _, lface1 = extract_faces(frame1, video1, scale_fact, False)
        _, lface2 = extract_faces(frame2, video2, scale_fact, False)
        if len(lface1) > 1 or len(lface2) > 1:
            print("Frame has more than one face!")
            err += 1
        if lface1.size == 0 or lface2.size == 0:
            print("Frame has none faces!")
            err += 1
            continue

        face1 = lface1[0]
        face2 = lface2[0]
        emb_face1 = get_embedding(face1)
        emb_face2 = get_embedding(face2)
        diff = g_embedder.compute_distance(emb_face1, emb_face2)
        plot_data['x'].append(i)
        plot_data['y'].append(diff)

        frame_cat = np.hstack((face1, face2)) 
        # cv2.imshow("Frame comparison", frame_cat)
        cv2.imwrite(result_path.joinpath(f"{j}.png"), frame_cat)
        j += 1

        if cv2.waitKey(1) == ord('q'):
            break
        i += 1
    print(f"Errors num: {err} for scale detection: {scale_fact}")
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    plot_data['x'] = np.asarray(plot_data['x'])
    plot_data['y'] = np.asarray(plot_data['y'])

    return plot_data


def process_videos():
    filenames = [
        "OrgUserNeutStas.mov",
        "OrgDronNeutStas.mov",
        "OrgUserEmotStas.mov",
        "OrgDronEmotStas.mov",
        "OrgUserNeutLukasz.mov",
        "OrgDronNeutLukasz.mov",
        "OrgUserEmotLukasz.mov",
        "OrgDronEmotLukasz.mov",
        "SpfUserNeutStas.mov",
        "SpfDronNeutStas.mov",
        "SpfUserEmotStas.mov",
        "SpfDronEmotStas.mov",
        "SpfUserNeutLukasz.mov",
        "SpfDronNeutLukasz.mov",
        "SpfUserEmotLukasz.mov",
        "SpfDronEmotLukasz.mov",
    ]
    videos = [Path().joinpath("input", "Videos", file) for file in filenames]

    # Pierwsza osoba w ciągu znaków odpowiada użytkownikowi
    standard_usage = {
        "StasNeut"   : compare_videos(videos[0], videos[1], 1.05, Path("results", "FrameComparison", "StandardUsage", "StasNeutral")),
        "StasEmot"   : compare_videos(videos[2], videos[3], 1.05, Path("results", "FrameComparison", "StandardUsage", "StasEmotion")),
        "LukaszNeut" : compare_videos(videos[4], videos[5], 1.05, Path("results", "FrameComparison", "StandardUsage", "LukaszNeutral")),
        "LukaszEmot" : compare_videos(videos[6], videos[7], 1.05, Path("results", "FrameComparison", "StandardUsage", "LukaszEmotion")),
    }
    naive_attack = {
        "StasNeut"   : compare_videos(videos[0], videos[5], 1.05, Path("results", "FrameComparison", "NaiveAttack", "StasNeutral")),
        "StasEmot"   : compare_videos(videos[2], videos[7], 1.05, Path("results", "FrameComparison", "NaiveAttack", "StasEmotion")),
        "LukaszNeut" : compare_videos(videos[4], videos[1], 1.05, Path("results", "FrameComparison", "NaiveAttack", "LukaszNeutral")),
        "LukaszEmot" : compare_videos(videos[6], videos[3], 1.05, Path("results", "FrameComparison", "NaiveAttack", "LukaszEmotion")),
    }
    spoof_norm_attack = {
        "StasNeut"   : compare_videos(videos[0], videos[0+8+1], 1.05, Path("results", "FrameComparison", "SpoofNormal", "StasNeutral")),
        "StasEmot"   : compare_videos(videos[2], videos[2+8+1], 1.05, Path("results", "FrameComparison", "SpoofNormal", "StasEmotion")),
        "LukaszNeut" : compare_videos(videos[4], videos[4+8+1], 1.05, Path("results", "FrameComparison", "SpoofNormal", "LukaszNeutral")),
        "LukaszEmot" : compare_videos(videos[6], videos[6+8+1], 1.05, Path("results", "FrameComparison", "SpoofNormal", "LukaszEmotion")),
    }
    spoof_perf_attack = {
        "StasNeut"   : compare_videos(videos[0], videos[0+8], 1.05, Path("results", "FrameComparison", "SpoofPerfect", "StasNeutral")),
        "StasEmot"   : compare_videos(videos[2], videos[2+8], 1.05, Path("results", "FrameComparison", "SpoofPerfect", "StasEmotion")),
        "LukaszNeut" : compare_videos(videos[4], videos[4+8], 1.05, Path("results", "FrameComparison", "SpoofPerfect", "LukaszNeutral")),
        "LukaszEmot" : compare_videos(videos[6], videos[6+8], 1.05, Path("results", "FrameComparison", "SpoofPerfect", "LukaszEmotion")),
    }
    for type, plot_data in standard_usage.items():
        np.savez_compressed(f'StandardUsage-{type}.npz', plot_data['x'], plot_data['y'])
    for type, plot_data in naive_attack.items():
        np.savez_compressed(f'NaiveAttack-{type}.npz', plot_data['x'], plot_data['y'])
    for type, plot_data in spoof_norm_attack.items():
        np.savez_compressed(f'SpoofNormal-{type}.npz', plot_data['x'], plot_data['y'])
    for type, plot_data in spoof_perf_attack.items():
        np.savez_compressed(f'SpoofPerfect-{type}.npz', plot_data['x'], plot_data['y'])


def create_animated_graphs():
    data_paths = [
        Path("results", "EmbDiff", "NaiveAttack-LukaszEmot.npz"),
        Path("results", "EmbDiff", "NaiveAttack-LukaszNeut.npz"),
        Path("results", "EmbDiff", "NaiveAttack-StasEmot.npz"),
        Path("results", "EmbDiff", "NaiveAttack-StasNeut.npz"),
        Path("results", "EmbDiff", "SpoofNormal-LukaszEmot.npz"),
        Path("results", "EmbDiff", "SpoofNormal-LukaszNeut.npz"),
        Path("results", "EmbDiff", "SpoofNormal-StasEmot.npz"),
        Path("results", "EmbDiff", "SpoofNormal-StasNeut.npz"),
        Path("results", "EmbDiff", "SpoofPerfect-LukaszEmot.npz"),
        Path("results", "EmbDiff", "SpoofPerfect-LukaszNeut.npz"),
        Path("results", "EmbDiff", "SpoofPerfect-StasEmot.npz"),
        Path("results", "EmbDiff", "SpoofPerfect-StasNeut.npz"),
        Path("results", "EmbDiff", "StandardUsage-LukaszEmot.npz"),
        Path("results", "EmbDiff", "StandardUsage-LukaszNeut.npz"),
        Path("results", "EmbDiff", "StandardUsage-StasEmot.npz"),
        Path("results", "EmbDiff", "StandardUsage-StasNeut.npz"),
    ]
    for data_path in data_paths:
        p1, p2 = data_path.stem.split('-')
        npdata = np.load(data_path)
        dat_x = npdata['arr_0']
        dat_y = npdata['arr_1']

        fig = plt.figure()
        l, = plt.plot([], [])
        plt.xlabel('Numer klatki nagrania')
        plt.ylabel('Odległość zakodowanych wektorów FaceNet')
        plt.title(f'Odległości wektorów FaceNet dla dwóch kolejnych klatek nagrania.\nTryb: "{p1}"; Użytkownik: "{p2}"')
        ax = plt.gca()
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0, 300)

        x_list = []
        y_list = []
        metadata = dict(title=data_path.stem, artist="264120")
        writer = FFMpegWriter(fps=30, metadata=metadata)
        with writer.saving(fig, f"{data_path.stem}.mp4", 100):
            for idx in range(len(dat_x)):
                x_list.append(dat_x[idx])
                y_list.append(dat_y[idx])
                l.set_data(x_list, y_list)
                writer.grab_frame()


if __name__ == "__main__":