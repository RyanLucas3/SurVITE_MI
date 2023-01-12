import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior() 

# Based on the paper MINE (Mutual Information Neural Estimation): https://arxiv.org/abs/1801.04062.
# MINE provides a NN approximation to Mutual Information for generic RVs X and Y.
# I adapt MINE to SurVITE: https://arxiv.org/pdf/2110.14001.pdf.
# Approximating the MI between the representation ɸ and the covariates X
# See research-task for details!

class SurVITE_MI:

    def __init__(self, α, H):
        '''
        :param α: the penalty term punishing information loss in the representation ɸ.
        :param H: the number of hidden layers in the network.
        '''

        # initialize network settings
        self.α = α
        self.H = H
                
    def SurVITE_MI_estimator(self, ɸ, X, A):

        '''
        :param ɸ: latent covariates for patients {1, ..., n} in mini-batch.
        :param X: observational (at-risk) covariates for patients {1, ..., n} in mini-batch.
        :param A: vector of assigned treatment for patients {1, ..., n} in mini-batch.
        :return: current MI loss and optimizer to be used within a tf loop.
        '''

        n = tf.shape(X)[0]
        
        # shuffle ɸ to obtain sample draws from the marginal
        ɸ_shuffle = tf.random_shuffle(ɸ)

        # Respective entries in the vertically aligned tensors 
        # correspond to the joint distribution and marginals
        ɸ_conc = tf.concat([ɸ, ɸ_shuffle], axis=0)
        X_conc = tf.concat([X, X], axis=0)
        
        # Forward-pass
        
        layerɸ = tf.compat.v1.layers.dense(inputs = ɸ_conc, units = self.H, activation = 'linear')
        layerX = tf.compat.v1.layers.dense(inputs = X_conc, units = self.H, activation = 'linear')
        layer2 = tf.nn.relu(layerɸ + layerX)
        output = tf.compat.v1.layers.dense(inputs = layer2, units = 1, activation = 'linear')
        
        # split f_ɸX and f_ɸ_X predictions
        f_ɸX = output[:n]
        f_ɸ_X = output[n:]
        
        # MI_1 loss: see research task equation 9
        MI_1 = tf.tensordot(f_ɸX, A, axes = 0) + tf.math.log(
            tf.tensordot(A, tf.math.exp(f_ɸ_X), axes = 0))

        # MI_0 loss: see research task equation 10
        MI_0 = tf.tensordot(f_ɸX, 1-A, axes = 0) + tf.math.log(
            tf.tensordot(1-A, tf.math.exp(f_ɸ_X), axes = 0))

        # L_MI: see research task equation 11
        # Minus because we're minimizing (we want to maximize MI or equivalently minimize -MI!)
        L_MI = tf.multiply(-1.0, MI_1 + MI_0)

        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(L_MI)

        return L_MI, train_step
