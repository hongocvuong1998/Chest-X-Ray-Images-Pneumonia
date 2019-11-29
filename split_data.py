import os
import numpy as np
from shutil import copy, move
import csv

def split_data_csv(path_csv, path_in, path_out):
    k=1
    with open(path_csv) as csvfile:
        readCSV = list(csv.reader(csvfile, delimiter=','))[1:]
        for row in readCSV:
            img_in = os.path.join(path_in, row[0].split(".")[0] + ".png")
            img_out = os.path.join(path_out, row[1], row[0].split(".")[0] + ".png")
            
            print(k," path_in  : ", img_in, "     path_out : ", img_out)
            move(img_in, img_out)
            k=k+1
        


def CreateTrainingEvaluation(folder):
	'''
	Input folder contain image data 
	Data:
		-0
		-1
		-2
	'''
	Eva=folder+'/Evaluation'
	Train=folder+'/Training'
	if not os.path.exists(Eva):
		os.makedirs(Eva)

    
	k=0
	print('Create Evaluation')
	for root, dirs , files in os.walk(folder):

		if k==0 or ('Evaluation' in root):
			k=1
			continue
		root_=root.split('/')

		pathdst=Eva+'/'+str(root_[-1:][0])

		if not os.path.exists(pathdst):
			os.makedirs(pathdst)


		listfiles = os.listdir(root)
		i = 0
		while  i <= len(listfiles)-5 :
			src=root +'/' + listfiles[i]
			dst=pathdst+'/' + listfiles[i]
			move(src, dst)
			i+=5
	print('Create Training')
	k=0
	for root, dirs , files in os.walk(folder):
		if k==0 or ('Evaluation' in root) or ('Training' in root):
			k=1
			continue
		root_=root.split('/')
		# pathdst=Train+'/'+str(root[-1:])
		pathdst=Train
		if not os.path.exists(pathdst):
			os.makedirs(pathdst)
		move(root, pathdst)


def CreateCSV(folder):
	'''
	Data contain Training and Evaluation
	Data:
		-Training:
			0
			1
			2
		Evaluation:
			0
			1
			2
	'''
	for root, dirs , files in os.walk(folder):
                for file in files:
                    EvaluationDataSet=open(folder+'/Evaluation.csv','a')
                    TrainingDataSet=open(folder+'/Training.csv', 'a')
                    
                    path=os.path.join(root, file)
                    
                    k=path.split('/')
                    string = path +' '+str(k[-2])
                    print(string)
                    if 'Evaluation' in string:
                        EvaluationDataSet.write(string)
                        EvaluationDataSet.write('\n')
                        EvaluationDataSet.close()
                    if 'Training' in string:
                        TrainingDataSet.write(string)
                        TrainingDataSet.write('\n')
                        TrainingDataSet.close()


if __name__ == '__main__':

	dataset ="./chest-xray-pneumonia"
	# split_data_csv(path_csv, path_in_log_specgram, path_out_log_specgram)
	# CreateTrainingEvaluation(path_out_log_specgram)
	CreateCSV(dataset)