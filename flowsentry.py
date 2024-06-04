import tensorflow as tf

class FlowSentry(tf.keras.Model):

    def __init__(self, **kwargs):
        super(FlowSentry, self).__init__(**kwargs)
        print("testing")
        self.time_distributed_layers = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        ])
        
        self.lstm1 = tf.keras.layers.LSTM(128)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.leak_output = tf.keras.layers.Dense(1, activation='sigmoid', name='leak_output')
        self.location_output = tf.keras.layers.Dense(1, activation='sigmoid', name='location_output')
        self.severity_output = tf.keras.layers.Dense(4, activation='softmax', name='severity_output')

    def call(self, inputs, training=None, mask=None):
        x = self.time_distributed_layers(inputs)
        x = self.lstm1(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        leak_output = self.leak_output(x)
        location_output = self.location_output(x)
        severity_output = self.severity_output(x)

        return {'leak_output': leak_output, 'location_output': location_output, 'severity_output': severity_output}

