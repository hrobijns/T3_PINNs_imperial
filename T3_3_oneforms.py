import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

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
# define the metric

    def g(self,x):
        x = x[0]
                
        g11 = 1.0
        g12 = 0.0
        g13 = 0.0
        g22 = 1.0
        g23 = 0.0
        g33 = 1.0
        
         
        # return the matrix
        g = tf.convert_to_tensor([
            [g11,g12,g13],
            [g12,g22,g23],
            [g13,g23,g33]
        ], dtype=tf.float64)
        return g

###################################################################################################
# define loss function, through defining hodge star and exterior derivative operators

    def evaluate(self, inputs): 
    # function which passes an input through the model
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
        outputs = self.model(inputs)
        return outputs
    
    def hodge_star(self, inputs, rank_acting_on, g):
    # define the Hodge operator, defined differently depending on which rank form it is acting on 
  
        if rank_acting_on == 1: # Hodge star acting on a 1-form: output is a 2-form
            u1, u2, u3 = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
            u = tf.concat([u1, -u2, u3], axis=1)
            sqrt_det_metric = tf.sqrt(tf.abs(tf.linalg.det(g)))
            inv_metric = tf.linalg.inv(g)
            product = sqrt_det_metric * tf.matmul(u, inv_metric)
            return tf.concat([product[:, 0:1], # coefficient of dx2 ^ dx3
                          product[:, 1:2], # coefficient of dx3 ^ dx1
                          product[:, 2:3]], # coefficient of dx1 ^ dx2
                          axis=1)
        
        elif rank_acting_on == 2: # Hodge star acting on a 2-form: output is a 1-form
            u1, u2, u3 = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
            u = tf.concat([u1, -u2, u3], axis=1)
            sqrt_det_metric = tf.sqrt(tf.abs(tf.linalg.det(g)))
            inv_metric = tf.linalg.inv(g)
            product = sqrt_det_metric * tf.matmul(u, inv_metric)
            return tf.concat([product[:, 0:1], # coefficient of dx1
                          product[:, 1:2], # coefficient of dx2
                          product[:, 2:3]], # coefficient of dx3
                          axis=1)
            
        elif rank_acting_on == 3: # Hodge star acting on a 3-form: output is a 0-form
            sqrt_det_metric = tf.sqrt(tf.abs(tf.linalg.det(g)))
            return sqrt_det_metric * inputs 
    
    def exterior_derivative(self, inputs, x, tape, rank_acting_on):
    # define the exterior derivative, defined differently depending on which rank form it is acting on
        
        if rank_acting_on == 0: # exterior derivative of a 0-form: output is a 1-form
            return tape.gradient(inputs, x) # coefficients of dx1, dx2 and dx3

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

    def calculate_PDE_error(self, x, tape):
        error = 0
        x = tf.expand_dims(x, axis=0)
        tape.watch(x)
        metric = self.g(x)
        u = self.model(x)
        f1,f2,f3 = tf.expand_dims(u[0, 0], axis=0), tf.expand_dims(u[0, 1], axis=0), tf.expand_dims(u[0, 2], axis=0)
        f1, f2, f3 = tf.expand_dims(f1, axis=-1), tf.expand_dims(f2, axis=-1),tf.expand_dims(f3, axis=-1)

        for i, f in enumerate([f1, f2, f3]):
            y = self.exterior_derivative(f, x, tape, rank_acting_on=0)
            
            # add 1 to the specific element of df based on the index i
            if i == 0:  # for f1, add 1 to the first element
                y = tf.concat([y[:, 0:1] + 1, y[:, 1:2], y[:, 2:3]], axis=1)
            elif i == 1:  # for f2, add 1 to the second element
                y = tf.concat([y[:, 0:1], y[:, 1:2] + 1, y[:, 2:3]], axis=1)
            elif i == 2:  # for f3, add 1 to the third element
                y = tf.concat([y[:, 0:1], y[:, 1:2], y[:, 2:3] + 1], axis=1)
            
            y = self.hodge_star(y, rank_acting_on=1, g=metric)
            y = self.exterior_derivative(y, x, tape, rank_acting_on=2)
            error += tf.square(self.hodge_star(y, rank_acting_on=3, g=metric))
        
        return error

    def loss(self, x_collocation, tape):
        PDE_errors = tf.vectorized_map(lambda x: self.calculate_PDE_error(x, tape), x_collocation)
        return tf.reduce_mean(PDE_errors)
    
###################################################################################################
# train network

    @tf.function
    def compute_loss_and_gradients(self, x_collocation):
        with tf.GradientTape(persistent=True) as tape:
            normalised_loss = self.loss(x_collocation, tape)
        grads = tape.gradient(normalised_loss, self.model.trainable_variables)
        return normalised_loss, grads

    def train(self, x_collocation, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            normalised_loss, grads = self.compute_loss_and_gradients(x_collocation)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {normalised_loss.numpy()}")

###################################################################################################
#  plot three one-forms on three axes T^3

    def plot_learned_1_forms(self):
        # set up the 3D cubes
        fig = plt.figure(figsize=(18, 6))
            
        # create a grid of points in the cube [0,1]x[0,1]x[0,1]
        num_points = 6
        x_vals = np.linspace(0, 1, num_points)
        y_vals = np.linspace(0, 1, num_points)
        z_vals = np.linspace(0, 1, num_points)
            
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # convert points to tensor
        points_tensor = tf.convert_to_tensor(points, dtype=tf.float64)

        # Get the outputs and their derivatives with respect to inputs
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(points_tensor)
            outputs = self.evaluate(points_tensor)
            
            f1_grads = tape.gradient(outputs[:, 0], points_tensor)
            f2_grads = tape.gradient(outputs[:, 1], points_tensor)
            f3_grads = tape.gradient(outputs[:, 2], points_tensor)

        f1_grads_adjusted = tf.concat([f1_grads[:, 0:1] + 1, f1_grads[:, 1:2], f1_grads[:, 2:]], axis=1)
        f2_grads_adjusted = tf.concat([f2_grads[:, 0:1], f2_grads[:, 1:2] + 1, f2_grads[:, 2:]], axis=1)
        f3_grads_adjusted = tf.concat([f3_grads[:, 0:1], f3_grads[:, 1:2], f3_grads[:, 2:3] + 1], axis=1)
            
        # define a function to plot a 3D vector field
        def plot_vector_field(ax, grads, title):
            U = grads[:, 0].numpy().reshape((num_points, num_points, num_points))
            V = grads[:, 1].numpy().reshape((num_points, num_points, num_points))
            W = grads[:, 2].numpy().reshape((num_points, num_points, num_points))
                
            ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])
            ax.set_title(title)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            
        # Plot the vector fields for f1, f2, and f3 derivatives
        ax1 = fig.add_axes([0.05, 0.15, 0.25, 0.7], projection='3d')  # [left, bottom, width, height]
        plot_vector_field(ax1, f1_grads_adjusted, "one-form (a)")

        ax2 = fig.add_axes([0.35, 0.15, 0.25, 0.7], projection='3d')
        plot_vector_field(ax2, f2_grads_adjusted, "one-form (b)")

        ax3 = fig.add_axes([0.65, 0.15, 0.25, 0.7], projection='3d')
        plot_vector_field(ax3, f3_grads_adjusted, "one-form (c)")
            
        plt.tight_layout()
        plt.show()

###################################################################################################
# running the program

if __name__ == '__main__':
    
    # generate collocation points
    num_samples = 1000
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    # initialise and train network
    pinn = PINN()
    pinn.train(x_collocation, epochs=500, learning_rate=0.001)

    # plot results and check for zeros
    pinn.plot_learned_1_forms()

