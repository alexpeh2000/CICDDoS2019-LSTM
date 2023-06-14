import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import resample
from sklearn import preprocessing

import os
dataset_path = []

for dirname, _, filenames in os.walk('D:/User/диплом ББСО-03-18/lstm/input'):
    for filename in filenames:
        if filename.endswith('.csv'):
            dfp = os.path.join(dirname, filename)
            dataset_path.append(dfp)



cols = list(pd.read_csv(dataset_path[1], nrows=1))

def load_file(path):
    # data = pd.read_csv(path, sep=',')
    data = pd.read_csv(path,
                   usecols =[i for i in cols if i != " Source IP" 
                             and i != ' Destination IP' and i != 'Flow ID' 
                             and i != 'SimillarHTTP' and i != 'Unnamed: 0'])

    return data

samples = pd.concat([load_file(dfp) for dfp in dataset_path], ignore_index=True)


import pandas as pd
import matplotlib.pyplot as plt

# Count the occurrences of each label
label_counts = samples[' Label'].value_counts()

# Create a bar plot to visualize the label counts
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Comparison of Label Column')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()




def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)

# Flows Packet/s e Bytes/s - Replace infinity by 0
samples = samples.replace('Infinity','0')
samples = samples.replace(np.inf,0)
#samples = samples.replace('nan','0')
samples[' Flow Packets/s'] = pd.to_numeric(samples[' Flow Packets/s'])

samples['Flow Bytes/s'] = samples['Flow Bytes/s'].fillna(0)
samples['Flow Bytes/s'] = pd.to_numeric(samples['Flow Bytes/s'])


#Label
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder to the label column and transform the labels
samples[' Label'] = label_encoder.fit_transform(samples[' Label'])

# Get the mapping between original labels and encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Print the mapping
for label, encoded_value in label_mapping.items():
    print(f"Label: {label} - Encoded Value: {encoded_value}")
    
#Timestamp - Drop day, then convert hour, minute and seconds to hashing 
colunaTime = pd.DataFrame(samples[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
samples[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
del colunaTime,stringHoras


print('Training data processed')


from keras.models import Sequential

from keras.layers import Dense,Embedding,Dropout,Flatten,MaxPooling1D,LSTM


def LSTM_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(LSTM(32,input_shape=(input_size,1), return_sequences=False))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(18, activation='softmax'))  # Update units and activation function

    print(model.summary())
    
    return model



def train_test(samples):
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Specify the data 
    X=samples.iloc[:,0:(samples.shape[1]-1)]
    
    # Specify the target labels and flatten the array
    #y= np.ravel(amostras.type)
    y= samples.iloc[:,-1]
    
    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = train_test(samples)


def format_3d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))


def compile_train(model,X_train,y_train,deep=True, epochs=10):
    
    if(deep==True):
        import matplotlib.pyplot as plt
#         optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.001)
#         model_lstm2.compile(optimizer=optimizer)

#         model.compile(loss='binary_crossentropy',
#                       optimizer='adam',
#                       metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train,epochs=epochs, batch_size=512, verbose=1)
        #model.fit(X_train, y_train,epochs=3)

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        print(model.metrics_names)
    
    else:
        model.fit(X_train, y_train) #SVM, LR, GD
    
    print('Model Compiled and Trained')
    return model


 def save_model(model,name):
    from keras.models import model_from_json
    
    arq_json = 'Models/' + name + '.json'
    model_json = model.to_json()
    with open(arq_json,"w") as json_file:
        json_file.write(model_json)
    
    arq_h5 = 'Models/' + name + '.h5'
    model.save_weights(arq_h5)
    print('Model Saved')
    
def load_model(name):
    from keras.models import model_from_json
    
    arq_json = 'Models/' + name + '.json'
    json_file = open(arq_json,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    arq_h5 = 'Models/' + name + '.h5'
    loaded_model.load_weights(arq_h5)
    
    print('Model loaded')
    
    return loaded_model
    






# UPSAMPLE 
    
X_train, X_test, y_train, y_test = train_test(samples)


#junta novamente pra aumentar o numero de normais
X = pd.concat([X_train, y_train], axis=1)

from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE(random_state=42).fit_resample(X, X[' Label'])
X_train = pd.DataFrame(X_train, columns=X_train.columns)

X_train=X_train.drop(' Label', axis=1)
input_size = (X_train.shape[1], 1)

del X
import pandas as pd
import matplotlib.pyplot as plt

# Count the occurrences of each label
label_counts = y_train.value_counts()

# Create a bar plot to visualize the label counts
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Comparison of Label Column')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


from keras.utils import to_categorical

# labels = to_categorical(samples[' Label'])
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


model_lstm2 = LSTM_model(82)
model_lstm2 = compile_train(model_lstm2,format_3d(X_train.astype(np.float32)),y_train.astype(np.float32), epochs=50)



############################################################################################################

!mkdir Models

!touch 'Models/lstm_50e.json'
save_model(model_lstm2,"lstm_50e")

del model_lstm2

!cp -r /kaggle/input/lstm-ddos-model/* /kaggle/working/Models/
!ls -lah Models

model_lstm = load_model('lstm_50e')
# model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


y_pred = model_lstm.predict(format_3d(X_test)) 

y_pred = y_pred.round()

from sklearn.metrics import accuracy_score
accuracy_score(y_test_label, y_pred_label)

from sklearn.metrics import classification_report
classification_report(y_test_label, y_pred_label)

# !apt -qqqq install aria2

# anngap file pcap sudah diextract menjadi csv
def detect_anomaly(csv_path, model):
    # read data
    data = pd.read_csv(csv_path,
               usecols =[i for i in cols if i != ' Destination IP' and i != 'Flow ID' 
                         and i != 'SimillarHTTP' and i != 'Unnamed: 0'])
    source_ip = data[' Source IP']
    data = data.drop(" Source IP", axis=1)
    
    # only for data from dataset
    data = data.drop(' Label', axis=1)
    
    # preprocess
    def string2numeric_hash(text):
        import hashlib
        return int(hashlib.md5(text).hexdigest()[:8], 16)

    # Flows Packet/s e Bytes/s - Replace infinity by 0
    data = data.replace('Infinity','0')
    data = data.replace(np.inf,0)
    #data = data.replace('nan','0')
    data[' Flow Packets/s'] = pd.to_numeric(data[' Flow Packets/s'])

    data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(0)
    data['Flow Bytes/s'] = pd.to_numeric(data['Flow Bytes/s'])

    #Timestamp - Drop day, then convert hour, minute and seconds to hashing 
    colunaTime = pd.DataFrame(data[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
    colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
    stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
    data[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
    del colunaTime,stringHoras

    predictions = model.predict(data)
    predictions = np.argmax(predictions, axis=1)

    # Create a DataFrame of anomalies
#     anomaly_df = pd.DataFrame()
#     for i in range(len(predictions)):
#         if predictions[i] != "0":
#             anomaly_df = anomaly_df.append({' Source IP': source_ip[i], "Label": predictions[i]}, ignore_index=True)

    anomaly_df = np.array([])
    for i in range(len(predictions)):
        if predictions[i] != 0:
            anomaly_df = np.append(anomaly_df, [source_ip[i], predictions[i]])
    
    # Print the IP address of the unique anomalies
#     print(anomaly_df["IP"].unique())
    anomaly_df = anomaly_df.reshape(-1, 2)
    label_map = {
        '0': "BENIGN",
        '1': "DrDoS_DNS",
        '2': "DrDoS_LDAP",
        '3': "DrDoS_MSSQL",
        '4': "DrDoS_NTP",
        '5': "DrDoS_NetBIOS",
        '6': "DrDoS_SNMP",
        '7': "DrDoS_UDP",
        '8': "LDAP",
        '9': "MSSQL",
        '10': "NetBIOS",
        '11': "Portmap",
        '12': "Syn",
        '13': "TFTP",
        '14': "UDP",
        '15': "UDP-lag",
        '16': "UDPLag",
        '17': "WebDDoS",
    }
    unique_values = np.unique(anomaly_df, axis=0)

    # Print the results in a fancy format.
    print('Source IP | anomaly')
    print('===================')
    for row in unique_values:
        print(f"{row[0]} | {label_map[row[1]]}")
detect_anomaly('D:/User/диплом ББСО-03-18/lstm/input/cicddos2019/DNS-testing.csv', model_lstm)