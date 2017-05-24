import numpy as np
import matplotlib.pyplot as plt
import edward as ed
import tensorflow as tf

from edward.models import Normal, Bernoulli, Laplace

#plt.style.use('ggplot')

def build_toy_dataset(N, w, noise_std=0.1):
  D = len(w)
  x = np.random.randn(N, D)*5
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  x = x.astype(np.float32)
  y = y.astype(np.float32)
  return x, y

N = 400
D = 1

w_true = np.random.randn(D)
print w_true
X_train, Y_train = build_toy_dataset(N, w_true)
X_test, Y_test = build_toy_dataset(N, w_true)


W_0 = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b_0 = Normal(loc=tf.zeros(1), scale=tf.ones(1))


X = tf.placeholder(tf.float32, shape=[N, D])
y = Normal(loc=ed.dot(X, W_0) + b_0, scale=tf.ones(N))


qW_0 = Normal(loc=tf.Variable(tf.random_normal([D])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

qb_0 = Normal(loc=tf.Variable(tf.random_normal([1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


mus = tf.stack(
    [ed.dot(X_train, qW_0.sample()) + qb_0.sample()
     for _ in range(10)])

sess = ed.get_session()
tf.global_variables_initializer().run()
outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0")

ax.plot(X_train[:], Y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(X_train[:], outputs[0], 'r.', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()


inference = ed.KLqp({W_0: qW_0, b_0: qb_0}, data={y: Y_train, X: X_train})
inference.run(n_iter=1000, n_samples=5)
outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0")

ax.plot(X_train[:], Y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(X_train[:], outputs[0], 'r.', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

print sess.run([qW_0.sample(), b_0.sample()])
