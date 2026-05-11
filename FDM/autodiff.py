import tensorflow as tf

# Define your function here, for example, a simple quadratic function
def function_to_minimize(x):
    return x ** 2

# Start with an initial value for x
x = tf.Variable(initial_value=3.0)

# Set the learning rate
learning_rate = 0.1

# Create an optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Run the gradient descent
num_iterations = 100
for _ in range(num_iterations):
    with tf.GradientTape() as tape:
        loss = function_to_minimize(x)
    
    gradients = tape.gradient(loss, x)
    optimizer.apply_gradients([(gradients, x)])

print("Optimized value of x:", x.numpy())
