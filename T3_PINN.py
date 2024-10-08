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
# define the metric.

    def g(self,x):
        x = x[0]
        
        g11 = (x[2]-0.5)**2
        g12 = 0
        g13 = 0
        g22 = 1.9*(x[0]-0.5)**2
        g23 = 0
        g33 = 1.9*(x[1]-0.5)**2
       
        '''
        g11 = 1
        g12 = 0
        g13 = 0
        g22 = 1
        g23 = 0
        g33 = 1
         '''

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
    # function which calculates overall laplacian of the 1-form, as the sum of two terms
        x = tf.expand_dims(x, axis=0)
        tape.watch(x)
        u = self.model(x)
        metric = self.g(x)        
        # 'LH_term': d*d* acting on the 1-form
        y = self.hodge_star(u, rank_acting_on=1, g=metric)
        y = self.exterior_derivative(y, x, tape, rank_acting_on=2)
        y = self.hodge_star(y, rank_acting_on=3, g=metric)
        LH_term = self.exterior_derivative(y, x, tape, rank_acting_on=0)

        # 'RH_term': *d*d acting on the 1-form
        y = self.exterior_derivative(u, x, tape, rank_acting_on=1)
        y = self.hodge_star(y, rank_acting_on=2, g=metric)
        y = self.exterior_derivative(y, x, tape, rank_acting_on=1)
        RH_term = self.hodge_star(y, rank_acting_on=2, g=metric)

        sum = LH_term + RH_term # sum of these terms is the laplacian of the one-form
        return tf.reduce_sum(tf.square(sum)) # return mean-squared error
      

    def loss(self, x_collocation, tape):
        PDE_errors = tf.vectorized_map(lambda x: self.calculate_PDE_error(x, tape), x_collocation)
        norm_factor = tf.reduce_sum(tf.abs(self.model(x_collocation)))
        loss = tf.reduce_mean(PDE_errors) / norm_factor
        return loss
    
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
#  simple plot on T^3 to visualise results

    def plot_learned_1_form(self, point):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # make the grid
        x, y, z = np.meshgrid(np.arange(0, 1.1, 0.2),
                              np.arange(0, 1.1, 0.2),
                              np.arange(0, 1.1, 0.2))
        grid_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        grid_tensor = tf.convert_to_tensor(grid_points, dtype=tf.float64)

        u = self.evaluate(grid_tensor).numpy()

        # Reshape to match grid
        u1 = u[:, 0].reshape(x.shape)*0.2
        u2 = u[:, 1].reshape(x.shape)*0.2
        u3 = u[:, 2].reshape(x.shape)*0.2
        ax.quiver(x, y, z, u1, u2, u3, normalize=False)

        ax.scatter(point[0], point[1], point[2], color='red', s=10)
        
        plt.show()

###################################################################################################
# simple check for zeroes: first a random check, then minimise from there.

    def find_zero_vector(self):
        # define the objective function for differential evolution
        def objective_function(x):
            x_tensor = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float64)
            g = self.g(x_tensor)
            u = self.evaluate(x_tensor)[0]
            u = tf.reshape(u, (3, 1))
            metric_norm_squared = tf.matmul(tf.matmul(tf.transpose(u), g), u)
            metric_norm = tf.sqrt(metric_norm_squared)
            return tf.squeeze(metric_norm).numpy()

        # set the bounds for the search space (unit cube)
        bounds = [(0, 1), (0, 1), (0, 1)]

        # perform Differential Evolution optimization
        result = differential_evolution(objective_function, bounds)

        # get the point with the smallest norm found
        min_norm = result.fun
        min_point = result.x

        # print the results
        print(f"Smallest norm found: {min_norm}")
        print(f"at point: {min_point}")
        print(f"verification: {objective_function(min_point)}")
        return min_point

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

    # check for zeroes
    zero = pinn.find_zero_vector()

    # plot results
    pinn.plot_learned_1_form(zero)