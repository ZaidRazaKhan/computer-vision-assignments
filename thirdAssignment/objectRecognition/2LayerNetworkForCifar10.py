import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


class CNN:
    def __init__(self):
        self.model = self.__define_model()

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")
    
    def __define_model(self):
        #defining model architecture
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # plot diagnostic learning curves
    def plot_graphs(self):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(self.history.history['loss'], color='blue', label='train')
        pyplot.plot(self.history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
        pyplot.plot(self.history.history['val_accuracy'], color='orange', label='test')
        # save plot to file
        filename = sys.argv[0].split('/')[-1]
        pyplot.savefig(filename + '_plot.png')
        pyplot.close()

    # run the test harness for evaluating a model
    def run_model(self, dataset):
        # load dataset
        trainX, trainY, testX, testY = dataset
        # fit model
        self.history = self.model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = self.model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))



class DataLoader:
    def __init__(self):
        (self.trainX, self.trainY), (self.testX, self.testY) = cifar10.load_data()
        self.trainY = to_categorical(self.trainY)
        self.testY = to_categorical(self.testY)

    def normalize_input_features(self):
        train_norm = self.trainX.astype('float32')
        test_norm = self.testX.astype('float32')
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        return train_norm, test_norm
    
    def get_labels(self):
        return self.trainY, self.testY


def main():
    cnn = CNN()
    dataloader = DataLoader()
    # load data
    trainX, testX = dataloader.normalize_input_features()
    trainY, testY  = dataloader.get_labels()
    # train and test model
    cnn.run_model((trainX, trainY, testX, testY))
    # save weights
    cnn.save_model()
    # plot learning curves
    cnn.plot_graphs()

    


if __name__ == '__main__':
    main()
    