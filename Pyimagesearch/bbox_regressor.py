from torch.nn import Dropout 
from torch.nn import Identity
from torch.nn import Linear 
from torch.nn import Module 
from torch.nn import ReLu
from torch.nn import Sequential 
from torch.nn import Sigmoid 

class ObjectDetector(Module):
    

    def __init__(self,baseModel,numClasses):
        super(ObjectDetector,self).__init__()

        self.baseModel=baseModel
        self.numClasses=numClasses

        self.regressor=Sequential(Linear(baseModel.fc.in_features,128),ReLu()
                                  Linear(128,64),ReLu(),
                                  Linear(64,32),ReLu(),
                                  Linear(32,4),Sigmoid())
        
        self.classifier = Sequential(
			Linear(baseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(), 
			Dropout(),
			Linear(512, self.numClasses)
		)

        self.baseModel.fc = Identity()


