from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def data_preprocessing1(  pathTrainL, pathLabelL):
    with open( pathTrainL ,'r') as f:
        for x in f:
            x,y = x.strip().split('\t')
            X_train.append(x)
            Y_train.append(y)

    category = []
    label = []
    dicMap = {}
    with open( pathLabelL,'r') as f:
        for x in f:
            x,y = x.strip().split('\t')
            category.append( y)
            label.append(x)

    st  = set( label )
    idd = 0
    for x in label:
        dicMap[x]= idd
        idd = idd + 1

    y_train = []
    for x in Y_train:
        y_train.append( dicMap[x])

    encoder = LabelEncoder()
    encoder.fit( Y_train)
    encoded_y = encoder.transform( Y_train)
    dummy_y = np_utils.to_categorical( encoded_y)

    return X_train, dummy_y, dicMap

    return X_train, y_train, dicMap