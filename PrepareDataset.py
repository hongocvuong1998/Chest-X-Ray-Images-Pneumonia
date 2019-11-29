from Header import *

def Create_dataset(path_data): 
    k=1
    label = []
    data = []

    list_label = os.listdir(path_data)
    for la in list_label:
        path_la = os.path.join(path_data, la)
        list_data = os.listdir(path_la)
        for da in list_data:
            path_da = os.path.join(path_la, da)
            my_data = np.load(path_da)
            data.append(my_data)
            label.append(la)

    label = np.asarray(label,dtype=np.int32)
    


    Dataset=datasets.TupleDataset(data, label)
    return Dataset

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path):
        self.base = chainer.datasets.LabeledImageDataset(path)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):  #    Train on image chanel=1
       
        # It reads the i-th image/label pair and return a preprocessed image.
        image, label = self.base[i]
        image=np.rollaxis(image, 0, 3)
        image = image[...,::-1] #BGR to RGB
        image = cv2.resize(image,(224, 224))  #(width,height)
        image = image.reshape(1,224,224)
        image *= (1.0 / 255.0) # Scale to [0, 1]
        image=np.asarray(image,dtype=np.float32)
        return image, label