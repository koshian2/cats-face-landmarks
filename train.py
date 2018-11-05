import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, History
import tensorflow.keras.backend as K
from keras.objectives import mean_squared_error
from PIL import Image
import numpy as np
import pickle, glob, random, os, zipfile
from tensorflow.contrib.tpu.python.tpu import keras_support

def enumerate_layers():
    # 確認用。サマリーとレイヤー名とindexの対応を調べる
    resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    resnet.summary()
    for i, layer in enumerate(resnet.layers):
        print(i, layer.name)

def create_resnet():
    # 転移学習用モデル
    resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for i in range(82):
        # res4a_branch2a（82）から訓練させる
        resnet.layers[i].trainable=False 

    x = GlobalAveragePooling2D()(resnet.output)
    # ランドマーク9×2点
    x = Dense(18, activation="sigmoid")(x)
    model = Model(resnet.inputs, x)
    return model

class CatGenerator:
    def __init__(self):
        with open("cats-dataset/cat_annotation.dat", "rb") as fp:
            self.annotation_data = pickle.load(fp)

    def flow_from_directory(self, batch_size, train=True, shuffle=True, use_data_augmentation=True):
        source_dir = "cats-dataset/train" if train else "cats-dataset/test"
        images = glob.glob(source_dir+"/*.jpg")
        X_cache, y_cache = [], []
        while True:
            if shuffle:
                np.random.shuffle(images)
            for img_path in images:
                with Image.open(img_path) as img:
                    width, height = img.size
                    img_array = np.asarray(img.resize((224, 224), Image.BILINEAR))
                basename = os.path.basename(img_path)
                data = self.annotation_data[basename]
                # アノテーションを0～1に変換
                annotation = np.zeros((9,2), dtype=np.float32)
                annotation[:, 0] = data[2][:, 0] / width
                annotation[:, 1] = data[2][:, 1] / height
                annotation = np.clip(annotation, 0.0, 1.0)

                if train and use_data_augmentation:
                    # 水平反転
                    if random.random() >= 0.5:
                        img_array = img_array[:, ::-1, :]
                        annotation[:, 0] = 1 - annotation[:, 0]
                        # 左目と右目の反転
                        annotation[0, :], annotation[1, :] = annotation[1, :], annotation[0, :].copy()
                        # 左耳と右耳の反転
                        annotation[3:6, :], annotation[6:9, :] = annotation[6:9, :], annotation[3:6, :].copy()
                    # PCA Color Augmentation
                    img_array = self.pca_color_augmentation(img_array)

                X_cache.append(img_array)
                y_cache.append(np.ravel(annotation))

                if len(X_cache) == batch_size:
                    X_batch = np.asarray(X_cache, dtype=np.float32) / 255.0
                    y_batch = np.asarray(y_cache, dtype=np.float32)
                    X_cache, y_cache = [], []
                    yield X_batch, y_batch

    def pca_color_augmentation(self, image_array_input):
        assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
        assert image_array_input.dtype == np.uint8

        img = image_array_input.reshape(-1, 3).astype(np.float32)
        img = (img - np.mean(img, axis=0)) / np.std(img, axis=0)

        cov = np.cov(img, rowvar=False)
        lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

        rand = np.random.randn(3) * 0.1
        delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
        delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

        img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
        return img_out


def loss_function_simple(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def loss_function_with_distance(y_true, y_pred):
    point_mse = mean_squared_error(y_true, y_pred)
    distance_mse = mean_squared_error(y_true[:, 2:18]-y_true[:, 0:16], y_pred[:, 2:18]-y_pred[:, 0:16])
    return point_mse + distance_mse

def loss_function_with_multiple_distance(y_true, y_pred):
    error = mean_squared_error(y_true, y_pred)
    for i in range(8):
        error += mean_squared_error(y_true[:, ((i+1)*2):18]-y_true[:, 0:(16-i*2)], y_pred[:, ((i+1)*2):18]-y_pred[:, 0:(16-i*2)])
    return error

# 三角形の面積を求める関数
def sarrus_formula(p1, p2, p3):
    # 座標シフト
    a = p2 - p1
    b = p3 - p1
    return K.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0]) / 2.0

from itertools import combinations

def loss_function_multiple_distance_and_triangle(y_true, y_pred):
    # 点損失
    error = mean_squared_error(y_true, y_pred)
    # 線の損失
    for i in range(8):
        error += mean_squared_error(y_true[:, ((i+1)*2):18]-y_true[:, 0:(16-i*2)], y_pred[:, ((i+1)*2):18]-y_pred[:, 0:(16-i*2)])
    # 面の損失
    for comb in combinations(range(9), 3):
        s_true = sarrus_formula(
            y_true[:, (comb[0]*2):(comb[0]*2+2)],
            y_true[:, (comb[1]*2):(comb[1]*2+2)],
            y_true[:, (comb[2]*2):(comb[2]*2+2)]
        )
        s_pred = sarrus_formula(
            y_pred[:, (comb[0]*2):(comb[0]*2+2)],
            y_pred[:, (comb[1]*2):(comb[1]*2+2)],
            y_pred[:, (comb[2]*2):(comb[2]*2+2)]
        )
        error += K.abs(s_true - s_pred)
    return error

def calc_area_loss(ear_true, ear_pred):
    left_x = K.expand_dims(K.min(ear_true[:, ::2], axis=-1))
    left_y = K.expand_dims(K.min(ear_true[:, 1::2], axis=-1))
    right_x = K.expand_dims(K.max(ear_true[:, ::2], axis=-1))
    right_y = K.expand_dims(K.max(ear_true[:, 1::2], axis=-1))
    # 予測のX,y
    pred_x = ear_pred[:, ::2]
    pred_y = ear_pred[:, 1::2]
    # ペナルティ
    penalty_x = K.maximum(left_x - pred_x, 0.0) + K.maximum(pred_x - right_x, 0.0)
    penalty_y = K.maximum(left_y - pred_y, 0.0) + K.maximum(pred_y - right_y, 0.0)
    return K.mean(penalty_x + penalty_y, axis=-1)

def loss_function_multiple_distance_and_area(y_true, y_pred):
    # 点損失
    error = mean_squared_error(y_true, y_pred)
    # 線の損失
    for i in range(8):
        error += mean_squared_error(y_true[:, ((i+1)*2):18]-y_true[:, 0:(16-i*2)], y_pred[:, ((i+1)*2):18]-y_pred[:, 0:(16-i*2)])
    # 右耳と左耳のエリア
    left_ear_true, left_ear_pred = y_true[:, 6:12], y_pred[:, 6:12]
    right_ear_true, right_ear_pred = y_true[:, 12:18], y_pred[:, 12:18]
    error += calc_area_loss(left_ear_true, left_ear_pred)
    error += calc_area_loss(right_ear_true, right_ear_pred)
    return error

class CatsCallback(Callback):
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.min_val_loss = np.inf

    def on_train_begin(self, logs):
        self.reset()

    def on_epoch_end(self, epoch, logs):
        if logs["val_loss"] < self.min_val_loss:
                self.model.save_weights("./cats_weights.hdf5", save_format="h5")
                self.min_val_loss = logs["val_loss"]
                print("Weights saved.", self.min_val_loss)

               
def train(batch_size, use_tpu, load_existing_weights):
    model = create_resnet()
    gen = CatGenerator()

    if load_existing_weights:
        model.load_weights("weights.hdf5")

    model.compile(tf.train.MomentumOptimizer(1e-3, 0.9), loss=loss_function_multiple_distance_and_area, metrics=[loss_function_simple])

    if use_tpu:
        tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    cb = CatsCallback(model)
    history = History()

    model.fit_generator(gen.flow_from_directory(batch_size, True), steps_per_epoch=6996//batch_size,
                        validation_data=gen.flow_from_directory(batch_size, False), validation_steps=2999//batch_size,
                        callbacks=[cb, history], epochs=200)

    with open("history.dat", "wb") as fp:
        pickle.dump(history.history, fp)

    with zipfile.ZipFile("cats_result.zip", "w") as zip:
        zip.write("history.dat")
        zip.write("cats_weights.hdf5")


if __name__ == "__main__":
    train(512, True, False)
