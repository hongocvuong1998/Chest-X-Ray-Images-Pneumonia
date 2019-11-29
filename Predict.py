from Header import *
# from PrepareDataset import *
from Model import *
import time
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib import cm

###################LOAD MODEL ####################
model = VGG16Net()
serializers.load_npz('/home/vuonghn/Project/Emotion-Recognition-Audio/Training_Image/Model_logs/model_epoch-500', model) #load file resule

def normalize(arr,min_,max_):
    return (arr-min_)/(max_ - min_)

def Probability(z):
	z_exp=[math.exp(i) for i in z]
	softmax=[i/sum(z_exp) for i in z_exp]
	return max(softmax)


def audio2image(path_audio):

	y, sr = librosa.load(path_audio,sr=12000)
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=512)
	log_S = librosa.power_to_db(S, ref=np.max)
	# log_S = librosa.amplitude_to_db(S)
	im = Image.fromarray(np.uint8(cm.seismic(normalize(log_S,-80,0))*255))
	arr = np.asarray(im.resize((512, 512),Image.ANTIALIAS))
	arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
	return arr

def predict_audio(path_audio):
	img = audio2image(path_audio)
	img=img.astype(np.float64)
	image = cv2.resize(img,(224, 224))
	image=np.rollaxis(image, 2, -3)
	image *= (1.0 / 255.0)
	image=np.asarray(image,dtype=np.float32)
	# start=time.time()
	y = model(image[None, ...])
	result= y.data.argmax(axis=1)[0]
	return result

def predict_audio_batch(batch_path):

	list_audio = os.listdir(batch_path) 
	total_video_test = len(list_audio)
	print("total_video_test ", total_video_test)

	numberPredict = 0


	with open('submit.csv', 'w') as csvfile:
		fieldnames = ['File','Label']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for audio in list_audio:
			path_audio = os.path.join(batch_path, audio)
			out_label = predict_audio(path_audio)
			writer.writerow({'File': audio,'Label': out_label})
			numberPredict = numberPredict + 1
			print("numberPredict: ", numberPredict ," / ",total_video_test )
predict_audio_batch("/home/vuonghn/Project/Emotion-Recognition-Audio/Public_test")



# path='D:\\Project\\CuocDuaSo2018\\Control\\Frame\\163.jpg'
# img=cv2.imread(path)
# cv2.imshow('img',img)

# DetectSign(img)

# # start=time.time()
# img=img.astype(np.float64)
# image = cv2.resize(img,(48, 48))
# image=np.rollaxis(image, 2, -3)
# image *= (1.0 / 255.0)
# image=np.asarray(image,dtype=np.float32)
# # start=time.time()
# y = model(image[None, ...])
# result= y.data.argmax(axis=1)[0]






