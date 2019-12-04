 
**Team members**:

Shilpa M - Monday batch (online)

Sushmitha M Katti - Friday batch (online)
<br/><br/>

**Base Model Accuracy** - 82.60
<br/>

**No Of Parameters used** - 85,701
**My Model Accuracy** - 81.56 (largest value)(48th Ephoch)
**Highest Accuracy Of My Model** -  
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

**EPHOCS**

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
1562/1562 [==============================] - 50s 32ms/step - loss: 1.3779 - acc: 0.5032 - val_loss: 1.3938 - val_acc: 0.5113
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
1562/1562 [==============================] - 44s 28ms/step - loss: 1.0207 - acc: 0.6420 - val_loss: 0.9229 - val_acc: 0.6787
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.8896 - acc: 0.6898 - val_loss: 0.9305 - val_acc: 0.6760
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.8030 - acc: 0.7216 - val_loss: 0.9114 - val_acc: 0.6886
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
1562/1562 [==============================] - 45s 28ms/step - loss: 0.7385 - acc: 0.7445 - val_loss: 0.9283 - val_acc: 0.6967
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.6942 - acc: 0.7599 - val_loss: 0.7167 - val_acc: 0.7493
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.6604 - acc: 0.7718 - val_loss: 0.7019 - val_acc: 0.7512
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.6298 - acc: 0.7807 - val_loss: 0.6927 - val_acc: 0.7637
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.6054 - acc: 0.7909 - val_loss: 0.6504 - val_acc: 0.7778
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
1562/1562 [==============================] - 44s 28ms/step - loss: 0.5825 - acc: 0.7978 - val_loss: 0.6697 - val_acc: 0.7729
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.5691 - acc: 0.8039 - val_loss: 0.7020 - val_acc: 0.7586
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.5501 - acc: 0.8092 - val_loss: 0.6352 - val_acc: 0.7854
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.5350 - acc: 0.8135 - val_loss: 0.6248 - val_acc: 0.7894
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.5243 - acc: 0.8188 - val_loss: 0.6377 - val_acc: 0.7866
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.5089 - acc: 0.8224 - val_loss: 0.6228 - val_acc: 0.7875
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.5051 - acc: 0.8236 - val_loss: 0.5950 - val_acc: 0.7982
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4901 - acc: 0.8305 - val_loss: 0.6143 - val_acc: 0.7901
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4833 - acc: 0.8328 - val_loss: 0.5980 - val_acc: 0.7986
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4741 - acc: 0.8357 - val_loss: 0.5830 - val_acc: 0.8030
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4643 - acc: 0.8370 - val_loss: 0.5937 - val_acc: 0.7998
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4573 - acc: 0.8390 - val_loss: 0.5719 - val_acc: 0.8044
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4498 - acc: 0.8421 - val_loss: 0.5963 - val_acc: 0.8014
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4476 - acc: 0.8425 - val_loss: 0.5524 - val_acc: 0.8110
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4395 - acc: 0.8465 - val_loss: 0.5771 - val_acc: 0.8034
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4318 - acc: 0.8475 - val_loss: 0.5518 - val_acc: 0.8114
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4311 - acc: 0.8495 - val_loss: 0.5859 - val_acc: 0.7992
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4252 - acc: 0.8513 - val_loss: 0.6073 - val_acc: 0.7974
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4184 - acc: 0.8551 - val_loss: 0.5949 - val_acc: 0.8009
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4149 - acc: 0.8563 - val_loss: 0.5698 - val_acc: 0.8088
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4082 - acc: 0.8584 - val_loss: 0.5657 - val_acc: 0.8111
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4080 - acc: 0.8570 - val_loss: 0.5772 - val_acc: 0.8089
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.4007 - acc: 0.8596 - val_loss: 0.5994 - val_acc: 0.8013
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3978 - acc: 0.8601 - val_loss: 0.5972 - val_acc: 0.7994
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3964 - acc: 0.8595 - val_loss: 0.5744 - val_acc: 0.8100
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3893 - acc: 0.8623 - val_loss: 0.5538 - val_acc: 0.8154
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3852 - acc: 0.8660 - val_loss: 0.5605 - val_acc: 0.8133
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3786 - acc: 0.8677 - val_loss: 0.5936 - val_acc: 0.8013
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3804 - acc: 0.8659 - val_loss: 0.5921 - val_acc: 0.8020
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3813 - acc: 0.8674 - val_loss: 0.5730 - val_acc: 0.8104
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3758 - acc: 0.8674 - val_loss: 0.5584 - val_acc: 0.8156
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3710 - acc: 0.8686 - val_loss: 0.5778 - val_acc: 0.8086
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3710 - acc: 0.8688 - val_loss: 0.5761 - val_acc: 0.8060
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3663 - acc: 0.8712 - val_loss: 0.5727 - val_acc: 0.8095
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3651 - acc: 0.8720 - val_loss: 0.5830 - val_acc: 0.8046
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3628 - acc: 0.8719 - val_loss: 0.5664 - val_acc: 0.8097
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3632 - acc: 0.8716 - val_loss: 0.5770 - val_acc: 0.8052
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3552 - acc: 0.8764 - val_loss: 0.5671 - val_acc: 0.8086
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3579 - acc: 0.8748 - val_loss: 0.5672 - val_acc: 0.8105
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3533 - acc: 0.8755 - val_loss: 0.5682 - val_acc: 0.8115
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
1562/1562 [==============================] - 45s 29ms/step - loss: 0.3560 - acc: 0.8732 - val_loss: 0.5791 - val_acc: 0.8087
Model took 2241.52 seconds to train


