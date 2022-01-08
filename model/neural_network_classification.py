import tensorflow as tf


class NeuralNetworkClassification:
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    model = None

    def __init__(self, batch_size, epochs):
        self.input_shape = (28, 28)
        self.num_classes = 10
        self.batch_size = batch_size
        self.epochs = epochs
        print('Tensorflow Version: {}'.format(tf.__version__))

    def loading_data(self):
        (self.X_train, self.Y_train), (self.X_val, self.Y_val) = tf.keras.datasets.mnist.load_data()
        print(
            f"X_train shape ={self.X_train.shape}, Y_train shape = {self.Y_train.shape}, X_val shape = {self.X_val.shape}, Y_val = {self.Y_val.shape}")

    def pre_processing_data(self):
        self.X_train = tf.cast(self.X_train, dtype=tf.float32)
        mean = tf.math.reduce_mean(self.X_train)
        std = tf.math.reduce_std(self.X_train)
        # reduce scale using the Z-Score function
        self.X_train = (self.X_train - mean) / std
        self.Y_train = (self.Y_train - mean) / std

    def model_building(self):
        tf.random.set_seed(3003)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(28, activation='relu'),
            tf.keras.layers.Dense(14, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Adding thuat toan cho mo hinh
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        # print the model parameters
        self.model.summary()
        # Adding the callback functions
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath="save_at_{epoch}.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            )
        ]

        hist = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2,
                              callbacks=callbacks)

    def evaluate_model(self):
        score = self.model.evaluate(self.X_val, self.Y_val, verbose=0)
        return score

    def __call__(self, *args, **kwargs):
        self.loading_data()
        self.pre_processing_data()
        self.model_building()
