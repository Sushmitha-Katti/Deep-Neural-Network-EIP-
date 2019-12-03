 
**Team members**:

Shilpa M - Monday batch (online)

Sushmitha M Katti - Friday batch (online)
<br/><br/>


**Model**

model = Sequential()

model.add(SeparableConv2D(32, kernel_size=(3, 3), strides=(1,1), input_shape=(32,32,3))) # o/p: 30 ,r:3

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(SeparableConv2D(64, kernel_size=(3, 3), strides=(1,1),kernel_regularizer=regularizers.l2(0.000001))) # o/p: 28 ,r:5

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(SeparableConv2D(128, kernel_size=(3, 3), strides=(1,1),kernel_regularizer=regularizers.l2(0.000001))) # o/p: 26 ,r:7

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(SeparableConv2D(256, kernel_size=(3, 3), strides=(1,1),kernel_regularizer=regularizers.l2(0.000001))) # o/p: 24 ,r:9

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(MaxPooling2D(pool_size=(2,2))) # o/p: 12 r:10

model.add(SeparableConv2D(128, kernel_size=(3, 3), strides=(1,1),kernel_regularizer=regularizers.l2(0.000001))) # o/p: 10 ,r:14

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(SeparableConv2D(10, kernel_size=(3, 3), strides=(1,1),kernel_regularizer=regularizers.l2(0.000001))) # o/p: 8 ,r:18

model.add(GlobalAveragePooling2D())

model.add(Activation('softmax'))

