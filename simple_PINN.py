# a simple PINN aimed at solving del^2(u)/del(x)^2 = sin(x)
################################################################################
# import relevant libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# training data generation: colocation points (at which physics loss is calculated) and boundary condition points to enforce agreement with b.c.s

def generate_colocation_points(x_lim_lower, x_lim_higher, number_points):
  x_colocation = np.random.uniform(low=x_lim_lower, high=x_lim_higher, size=(number_points, 1)).astype(np.float32)
  x_colocation_tf = tf.convert_to_tensor(x_colocation, dtype=tf.float32)
  return x_colocation_tf

def generate_boundary_points():
    # boundary points for u(x=0)=0
    x_boundary1 = np.full((1, 1), 0.0).astype(np.float32)
    u_boundary1 = np.full((1, 1), 0.0).astype(np.float32)

    # boundary points for u(x=3)=2.859
    x_boundary2 = np.full((1, 1), 3.0).astype(np.float32)
    u_boundary2 = np.full((1, 1), 2.859).astype(np.float32)

    # concatenate the boundary points
    x_boundary = np.concatenate([x_boundary1, x_boundary2], axis=0)
    u_boundary = np.concatenate([u_boundary1, u_boundary2], axis=0)

    # convert to tensors
    x_boundary_tf = tf.convert_to_tensor(x_boundary, dtype=tf.float32)
    u_boundary_tf = tf.convert_to_tensor(u_boundary, dtype=tf.float32)

    return x_boundary_tf, u_boundary_tf

################################################################################
# define PINN
class PINN(tf.keras.Model):
  def __init__(self):
    super().__init__() # to access features of the parent class, tf.keras.Model     
    self.dense1 = tf.keras.layers.Dense(16, activation='tanh')
    self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
    self.dense3 = tf.keras.layers.Dense(16, activation='tanh')
    self.output_layer = tf.keras.layers.Dense(1, activation=None)
  
  def call(self, inputs): # passes the inputs through the architecture defined above
    hidden1 = self.dense1(inputs)
    hidden2 = self.dense2(hidden1)
    hidden3 = self.dense3(hidden2)
    output = self.output_layer(hidden3)
    return output

  def compute_gradients(self, x): # computes required gradients for minimising loss
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      u_pred = self(x)
      u_x = tape.gradient(u_pred, x)
      u_xx = tape.gradient(u_x, x)
    del tape
    return u_xx

# define loss function
def loss(model, x_colocation, x_boundary, u_boundary):
    # compute loss at colocation points
    u_xx = model.compute_gradients(x_colocation)
    pde_loss = u_xx - tf.sin(x_colocation)
    physics_loss = tf.reduce_mean(tf.square(pde_loss))

    # compute loss at boundary points
    u_boundary_pred = model(x_boundary)
    boundary_loss = tf.reduce_mean(tf.square(u_boundary_pred - u_boundary))

    return physics_loss + boundary_loss # scale if necessary

################################################################################
# define training of the model
def train_network(x_colocation_tf, x_boundary_tf, u_boundary_tf, model=PINN(), num_epochs=501):
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  
  for epoch in range(num_epochs):
      with tf.GradientTape() as tape:
          loss_value = loss(model, x_colocation_tf, x_boundary_tf, u_boundary_tf)
  
      gradients = tape.gradient(loss_value, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
      if epoch % 100 == 0:
          print(f"Epoch {epoch}/{num_epochs-1}, Loss: {loss_value.numpy()}")
  return model

################################################################################
# visualisation of the solution
def analytical_solution(x):
  return x - np.sin(x) 

def plot_PINN_prediction(x_lim_lower, x_lim_higher, model):
  x_test = np.linspace(x_lim_lower, x_lim_higher, 300).reshape(-1, 1).astype(np.float32)
  x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
  u_pred = model(x_test_tf)
  plt.plot(x_test, analytical_solution(x_test), linestyle = ":", label='analytical')
  plt.plot(x_test, u_pred.numpy(), alpha=0.5, label='PINN predicted solution')
  plt.scatter(x_boundary_tf.numpy(), u_boundary_tf.numpy(), color='red', s = 8, label='b.c. points')
  plt.xlim(x_lim_lower,x_lim_higher)
  plt.xlabel('x')
  plt.ylabel('u')
  plt.legend()
  plt.show()

################################################################################
# running the program:
if __name__ == '__main__':
  x_colocation_tf = generate_colocation_points(0, 5, 1000)
  x_boundary_tf, u_boundary_tf = generate_boundary_points()
  trained_model = train_network(x_colocation_tf, x_boundary_tf, u_boundary_tf, PINN())
  plot_PINN_prediction(0, 5, trained_model)
