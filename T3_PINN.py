import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###################################################################################################
# set the floating point precision
tf.keras.backend.set_floatx('float64')

###################################################################################################
# define a trig activitation layer - this enforces periodicity, 
# meaning the solution lies on the manifold T^3, as requested.

class TrigActivation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2*np.pi*inputs), tf.cos(2*np.pi*inputs)], axis=1)

###################################################################################################
# define the architecture of the NN 

class PINN:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,)),
            TrigActivation(), # inputs mapped to periodic space
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=64, activation='tanh'),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=3) # various hidden layers
        ])
        
###################################################################################################
# define the metric, with an x-dependance
    def g(self, x):
        g = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return tf.convert_to_tensor(g, dtype=tf.float64)

###################################################################################################
# define loss function, through defining hodge star and exterior derivative operators

    def evaluate(self, inputs): 
    # function which passes an input through the model
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
        outputs = self.model(inputs)
        return outputs
    
    def hodge_star(self, inputs, x, rank_acting_on):
    # define the Hodge operator, defined differently depending on which rank form it is acting on 
        sqrt_det_metric = tf.sqrt(tf.abs(tf.linalg.det(self.g(x))))
        inv_metric = tf.linalg.inv(self.g(x))
        
        if rank_acting_on == 1: # Hodge star acting on a 1-form: output is a 2-form
            product = sqrt_det_metric * tf.matmul(inputs, inv_metric)
            return tf.concat([product[:, 0:1], # coefficient of dx2 ^ dx3
                          product[:, 1:2], # coefficient of dx3 ^ dx1
                          product[:, 2:3]], # coefficient of dx1 ^ dx2
                          axis=1)
        
        elif rank_acting_on == 2: # Hodge star acting on a 2-form: output is a 1-form
            product = sqrt_det_metric * tf.matmul(inputs, inv_metric)
            return tf.concat([product[:, 0:1], # coefficient of dx1
                          product[:, 1:2], # coefficient of dx2
                          product[:, 2:3]], # coefficient of dx3
                          axis=1)
            
        elif rank_acting_on == 3: # Hodge star acting on a 3-form: output is a 0-form
            return (1/sqrt_det_metric) * inputs 

    def exterior_derivative(self, inputs, x, tape, rank_acting_on):
    # define the exterior derivative, defined differently depending on which rank form it is acting on

        if rank_acting_on == 0: # exterior derivative of a 0-form: output is a 3-form
            return tape.gradient(inputs, x)

        elif rank_acting_on == 1: # exterior derivative of a 1-form: output is a 2-form
            derivatives = tf.concat([
                tape.gradient(inputs[:, 0], x),
                tape.gradient(inputs[:, 1], x),
                tape.gradient(inputs[:, 2], x),
            ], axis=1)
        
            return tf.concat([
            tf.expand_dims(derivatives[:, 5] - derivatives[:, 7], axis=1), # coefficient of dx2 ^ dx3 
            tf.expand_dims(derivatives[:, 6] - derivatives[:, 2], axis=1), # coefficient of dx3 ^ dx1
            tf.expand_dims(derivatives[:, 1] - derivatives[:, 3], axis=1) # coefficient of dx1 ^ dx2 
        ], axis=1)
        
        elif rank_acting_on == 2: # exterior derivative of a 2-form: output is a 3-form
            derivatives = tf.concat([
                tape.gradient(inputs[:, 0], x),
                tape.gradient(inputs[:, 1], x),
                tape.gradient(inputs[:, 2], x),
            ], axis=1) 

            return tf.expand_dims(derivatives[:, 0] - derivatives[:, 4] + derivatives[:, 8], axis=1)

    def calculate_PDE_error(self, x):
    # function which calculates overall laplacian of the 1-form, as the sum of two terms

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.model(x)
            
            # 'LH_term': d*d* acting on the 1-form
            y = self.hodge_star(u, x, rank_acting_on=1)
            y = self.exterior_derivative(y, x, tape, rank_acting_on=2)
            y = self.hodge_star(y, x, rank_acting_on=3)
            LH_term = self.exterior_derivative(y, x, tape, rank_acting_on=0)

            # 'RH_term': *d*d acting on the 1-form
            y = self.exterior_derivative(u, x, tape, rank_acting_on=1)
            y = self.hodge_star(y, x, rank_acting_on=2)
            y = self.exterior_derivative(y, x, tape, rank_acting_on=1)
            RH_term = self.hodge_star(y,x, rank_acting_on=2)

        del tape
        return LH_term + RH_term # sum of these terms is the laplacian of the one-form

    def loss(self, x_collocation):
        PDE_error = self.calculate_PDE_error(x_collocation)
        loss = tf.reduce_mean(tf.square(PDE_error)) # we want laplacian of the one-form to be zero to solve PDE
        norm_factor = tf.reduce_sum(tf.abs(self.model(x_collocation)))
        normalised_loss = loss / norm_factor # this is here so that the NN does not learn zero
        return normalised_loss

###################################################################################################
# train network

    def train(self, x_collocation, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                normalised_loss = self.loss(x_collocation)
            grads = tape.gradient(normalised_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {normalised_loss.numpy()}")

###################################################################################################
#  simple plot on T^3 to visualise results

    def plot_learned_1_form(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = np.meshgrid(np.arange(0, 1.1, 0.2),
                              np.arange(0, 1.1, 0.2),
                              np.arange(0, 1.1, 0.2))
        grid_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        grid_tensor = tf.convert_to_tensor(grid_points, dtype=tf.float64)

        u = self.evaluate(grid_tensor).numpy()

        u1 = u[:, 0].reshape(x.shape)
        u2 = u[:, 1].reshape(x.shape)
        u3 = u[:, 2].reshape(x.shape)

        ax.quiver(x, y, z, u1, u2, u3, length=0.1, normalize=True)

        plt.show()

###################################################################################################
#  running the program

if __name__ == '__main__':
    
    # generate collocation points
    num_samples = 1000
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    # initialise and train network
    pinn = PINN()
    pinn.train(x_collocation, epochs=100, learning_rate=0.001)
    
    # check for periodicity (should always be true)
    inputs = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
    outputs = pinn.evaluate(inputs)
    print(f"Outputs for [1, 1, 1] and [2, 2, 2]:\n{outputs.numpy()} (these should be exactly the same if the NN has learnt a periodic solution)")

    # check for constant output (should be true for the identity metric)
    random_inputs = np.random.uniform(low=0, high=1, size=(5, 3))
    random_inputs_tensor = tf.convert_to_tensor(random_inputs, dtype=tf.float64)
    random_outputs = pinn.evaluate(random_inputs_tensor).numpy()
    print("Random inputs:")
    print(random_inputs)
    print("Random outputs:")
    print(random_outputs)
    print("(these 5 output vectors should be pretty much the same if the identity metric is used)")

    # plot results
    pinn.plot_learned_1_form()