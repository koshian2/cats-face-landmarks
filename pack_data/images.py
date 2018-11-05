import glob, shutil, os
from sklearn.model_selection import train_test_split

def make_dataset():
    train_dir = "cats-dataset/train"
    test_dir = "cats-dataset/test"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    images = glob.glob("cats/*/*.jpg")
    img_train, img_test = train_test_split(images, test_size=0.3)
    for img in img_train:
        print(img)
        basename = os.path.basename(img)
        shutil.copy(img, train_dir+"/"+basename)
    for img in img_test:
        print(img)
        basename = os.path.basename(img)
        shutil.copy(img, test_dir+"/"+basename)

make_dataset() # 9995件
