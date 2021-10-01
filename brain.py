from imageai.Prediction import ImagePrediction 
import os 
execution_path=os.getcwd() #pwd present wokring directory

prediction = ImagePrediction() #instanciate  
prediction.setModelTypeAsMobileNetV2() #what model we want to use like resnet,squeeze net,inceoption,densenet
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "giraffe.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities): #prediction = what it is, probailitiy is how likely it is.
    print(eachPrediction , " : " , eachProbability)