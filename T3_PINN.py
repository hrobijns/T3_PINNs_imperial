import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# set the floating point precision
tf.keras.backend.set_floatx('float64')

class TrigActivation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2*np.pi*inputs), tf.cos(2*np.pi*inputs)], 1)

class PINN:
    def __init__(self):
    # define the neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,)),  # input layer with 3 inputs, the 3 elements of x, the input point in R^3: x1, x2, x3
            TrigActivation(), # enforcing periodicity, ensuring model is working on T^3 
            tf.keras.layers.Dense(units=32, activation='tanh'), # hidden layers of the PINN
            tf.keras.layers.Dense(units=64, activation='tanh'),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=3)  # output layer for f1(x), f2(x), f3(x)
        ])

    def star_d_2_form(self, derivative):
    # define (hodge star . d) acting on a 2-form
        output = tf.concat([
            derivative[:, 3:4] - derivative[:, 5:6],   # coefficient of dx1 (del(f_2)/del(x_3)-del(f_3)/del(x_2))
            derivative[:, 4:5] - derivative[:, 1:2],   # coefficient of dx2 (del(f_3)/del(x_1)-del(f_1)/del(x_3))
            derivative[:, 0:1] - derivative[:, 2:3]    # coefficient of dx3 (del(f_1)/del(x_2)-del(f_2)/del(x_1))
        ], axis=1)
        return output
    
    def star_d_star_1_form(self, derivative):
    # define (d . hodge star . d) acting on a 1-form
        return derivative[:, 0] + derivative[:, 1] + derivative[:, 2] # this simple form can be found by working through the maths
    
    def d_0_form(self, derivative):
    # define d acting on a 0-form
        output = tf.concat([
            derivative[:, 0:1],  # coefficient of dx1
            derivative[:, 1:2],  # coefficient of dx2
            derivative[:, 2:3]   # coefficient of dx3
        ], axis=1)
        return output

    def calculate_PDE(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.model(x)
            du_dx = tape.gradient(u, x)
            
            u_prime = self.star_d_2_form(du_dx)
            du_prime_dx = tape.gradient(u_prime, x)
            RH_term = self.star_d_2_form(du_prime_dx)
            
            alpha = self.star_d_star_1_form(du_dx)
            alpha_grad = tape.gradient(alpha, x)
            LH_term = self.d_0_form(alpha_grad)

        del tape    
        return RH_term, LH_term

    def loss(self, x_collocation):
        RH_term, LH_term = self.calculate_PDE(x_collocation)
        loss = tf.reduce_mean(tf.square(RH_term + LH_term))
        norm_factor = tf.reduce_sum(tf.abs(self.model(x_collocation)))
        normalised_loss = loss / norm_factor # this term is currently here so it doesn't learn zero
        return normalised_loss

    def train(self, x_collocation, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                normalised_loss = self.loss(x_collocation)
            grads = tape.gradient(normalised_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) 
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {normalised_loss.numpy()}")
        
    def evaluate(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
        outputs = self.model(inputs)
        return outputs

if __name__ == '__main__': 
    # generate collocation points within a unit cube
    num_samples = 1000
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    # initialise PINN model
    pinn = PINN()
    
    # train the model
    pinn.train(x_collocation, epochs = 1000, learning_rate = 0.001)

    # periodicity verification
    inputs = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
    outputs = pinn.evaluate(inputs)
    print(f"Outputs for [1, 1, 1] and [2, 2, 2]:\n{outputs.numpy()}")

    # verify whether it has learnt a constant
    random_inputs = np.random.uniform(low=0, high=1, size=(5, 3))
    random_inputs_tensor = tf.convert_to_tensor(random_inputs, dtype=tf.float64)
    random_outputs = pinn.evaluate(random_inputs_tensor).numpy()
    resulting_arrays = random_outputs / random_inputs
    print("Resulting arrays from element-wise division of outputs by inputs:")
    print(resulting_arrays) # each element across the arrays (i.e. the second element in each) should be similar if it has learnt a constant