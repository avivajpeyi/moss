import timeit
import numpy as np
from scipy.sparse import coo_matrix

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class SpecPrep:  # Parent used to create SpecModel object
    def __init__(self, x):
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # ts:     time series x
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.ts = x
        if x.shape[1] < 2:
            raise Exception("Time series should be at least 2 dimensional.")

        self.y_ft = []    # inital
        self.freq = []
        self.p_dim = []
        #self.Xmat = []
        self.Zar = []
        # other self variables can be defined later within methods
        # in init, can't use the methods in the class defined below
        # 需要逐层定义，不能跨级，例如self.a.b不可行，必须先定义有b属性的a才可以
    
    # scaled fft and get the elements of freq = 1:[Nquist]
    # discarding the rest of freqs
    def sc_fft(self):
        # x is a n-by-p matrix
        # unscaled fft
        x = self.ts
        num_segments = self.nchunks
        len_chunk = x.shape[0] // num_segments
        x = np.array(np.split(x[0:len_chunk*num_segments,:], num_segments))
        
        y = []
        for i in range(num_segments):
            y_fft = np.apply_along_axis(np.fft.fft, 0, x[i])
            y.append(y_fft)
        y = np.array(y)
        
        # scale it
        n = x.shape[1]
        y = y / np.sqrt(n)  #np.sqrt(n)
        # discard 0 freq
        
        Ts = 1 
        fq_y = np.fft.fftfreq(np.size(x,1), Ts)
        
        if np.mod(n, 2) == 0:
            # n is even
            y = y[:, 0:int(n/2) , :]
            fq_y = fq_y[0:int(n/2)]
        else:
            # n is odd
            y = y[:, 0:int((n-1)/2), :]
            fq_y = fq_y[0:int((n-1)/2 )]
        p_dim = x.shape[2]
            
        self.y_ft = y
        self.freq = fq_y
        self.p_dim = p_dim
        self.num_obs = fq_y.shape[0]
        return dict(y=y, fq_y=fq_y, p_dim=p_dim)




    # Demmler-Reinsch basis for linear smoothing splines (Eubank,1999)
    def DR_basis(self, N = 10):
        # nu: vector of frequences
        # N:  amount of basis used
        # return a len(nu)-by-N matrix
        nu = self.freq
        basis = np.array([np.sqrt(2)*np.cos(x*np.pi*nu*2) for x in np.arange(1, N + 1)]).T
        return basis
    #  DR_basis(y_ft$fq_y, N=10)


    # cbinded X matrix 
    def Xmtrix(self, N_delta = 15, N_theta=15):
        nu = self.freq
        X_delta = np.concatenate([np.column_stack([np.repeat(1, nu.shape[0]), nu]), self.DR_basis(N = N_delta)], axis = 1)
        X_theta = np.concatenate([np.column_stack([np.repeat(1, nu.shape[0]), nu]), self.DR_basis(N = N_theta)], axis = 1)
        try:
            if self.Xmat_delta is not None:
                  Xmat_delta = tf.convert_to_tensor(X_delta, dtype = tf.float32)
                  Xmat_theta = tf.convert_to_tensor(X_theta, dtype = tf.float32)
                  return Xmat_delta, Xmat_theta
        except: # NPE
            self.Xmat_delta = tf.convert_to_tensor(X_delta, dtype = tf.float32) # basis matrix
            self.Xmat_theta = tf.convert_to_tensor(X_theta, dtype = tf.float32)
            self.N_delta = N_delta # N
            self.N_theta = N_theta
            return self.Xmat_delta, self.Xmat_theta


    def set_y_work(self):
        y_work = self.y_ft
        self.y_work = y_work
        return y_work
    
    def dmtrix_k(self, y_k):
        
        n, p_work = y_k.shape
        Z_k = np.zeros([n, p_work, int(p_work*(p_work - 1)/2)], dtype = complex )
        
        for j in range(n):
            count = 0
            for i in np.arange(1, p_work):
                Z_k[j, i, count:count+i] = y_k[j, :i]#.flatten()
                count += i
        return Z_k
    
    def Zmtrix(self): # dense Z matrix
        y_work = self.set_y_work()
        c, n, p = y_work.shape
        if p > 1:
            if c == 1:
                y_ls = np.squeeze(y_work, axis=0)
                Z_ = self.dmtrix_k(y_ls)
            else:
                y_ls = np.squeeze(np.split(y_work, c))
                Z_ = np.array([self.dmtrix_k(x) for x in y_ls])
        else:
            Z_ = 0
        self.Zar_re = np.real(Z_) # add new variables to self, if Zar not defined in init at the beginning
        self.Zar_im = np.imag(Z_)
        return self.Zar_re, self.Zar_im
    
    
    # Sparse matrix form of Zmtrix()
    def SparseZmtrix(self): # sparse Z matrix
        y_work = self.y_work()
        n, p = y_work.shape
        
        if p == 1:
            raise Exception('To use sparse representation, dimension of time series should be at least 2')
            return
        
        y_ls = np.split(y_work, n)
        
        coomat_re_ls = []
        coomat_im_ls = []
        for i in range(n):
            Zar = self.dmtrix_k(y_ls[i])
            Zar_re = np.real(Zar)
            Zar_im = np.imag(Zar)
            coomat_re_ls.append(coo_matrix(Zar_re))
            coomat_im_ls.append(coo_matrix(Zar_im))
        
        Zar_re_indices = []
        Zar_im_indices = []
        Zar_re_values = []
        Zar_im_values = []
        for i in range(len(coomat_re_ls)):            
            Zar_re_indices.append(np.stack([coomat_re_ls[i].row, coomat_re_ls[i].col], -1))
            Zar_im_indices.append(np.stack([coomat_im_ls[i].row, coomat_im_ls[i].col], -1))
            Zar_re_values.append(coomat_re_ls[i].data)
            Zar_im_values.append(coomat_im_ls[i].data)
        
        
        self.Zar_re_indices = Zar_re_indices
        self.Zar_im_indices = Zar_im_indices
        self.Zar_re_values = Zar_re_values
        self.Zar_im_values = Zar_im_values
        self.Zar_size = Zar.shape
        return [self.Zar_re_indices, self.Zar_re_values], [self.Zar_im_indices, self.Zar_im_values]

# 
## Spectrum Model, subclass of SpecPrep 继承了inital以及所有的methods
#
class SpecModel(SpecPrep):
    def __init__(self, x, hyper, sparse_op=False, nchunks=1):
        super().__init__(x)
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # hyper:  list of hyperparameters for prior
        # ts:     time series == x
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.hyper = hyper
        self.sparse_op = sparse_op
        self.trainable_vars = []   # all trainable variables
        self.nchunks = nchunks
    
    def toTensor(self):
        # convert to tensorflow object
        self.ts = tf.convert_to_tensor(self.ts, dtype = tf.float32)
        self.y_ft = tf.convert_to_tensor(self.y_ft, dtype = tf.complex64)
        self.y_work = tf.convert_to_tensor(self.y_ft, dtype = tf.complex64)
        self.y_re = tf.math.real(self.y_work) # not y_ft
        self.y_im = tf.math.imag(self.y_work)
        self.freq = tf.convert_to_tensor(self.freq, dtype = tf.float32)
        self.p_dim = tf.convert_to_tensor(self.p_dim, dtype = tf.int32)
        self.N_delta = tf.convert_to_tensor(self.N_delta, dtype = tf.int32)
        self.N_theta = tf.convert_to_tensor(self.N_theta, dtype = tf.int32)
        self.Xmat_delta = tf.convert_to_tensor(self.Xmat_delta, dtype = tf.float32)
        self.Xmat_theta = tf.convert_to_tensor(self.Xmat_theta, dtype = tf.float32)        
        
        if self.sparse_op == False:
            self.Zar = tf.convert_to_tensor(self.Zar, dtype = tf.complex64)  # complex array
            self.Z_re = tf.convert_to_tensor(self.Zar_re, dtype = tf.float32)
            self.Z_im = tf.convert_to_tensor(self.Zar_im, dtype = tf.float32)
        else: # sparse_op == True
            self.Zar_re_indices = [tf.convert_to_tensor(x, tf.int64) for x in self.Zar_re_indices] # int64 required by tf.sparse.SparseTensor
            self.Zar_im_indices = [tf.convert_to_tensor(x, tf.int64) for x in self.Zar_im_indices]
            self.Zar_re_values = [tf.convert_to_tensor(x, tf.float32) for x in self.Zar_re_values]
            self.Zar_im_values = [tf.convert_to_tensor(x, tf.float32) for x in self.Zar_im_values]
            
            self.Zar_size = tf.convert_to_tensor(self.Zar_size, tf.int64)
    
            self.Z_re = [tf.sparse.SparseTensor(x, y, self.Zar_size) for x, y in zip(self.Zar_re_indices, self.Zar_re_values)]
            self.Z_im = [tf.sparse.SparseTensor(x, y, self.Zar_size) for x, y in zip(self.Zar_im_indices, self.Zar_im_values)]


        self.hyper = [tf.convert_to_tensor(self.hyper[i], dtype = tf.float32) for i in range(len(self.hyper))]
        if self.p_dim > 1:
            self.n_theta = tf.cast(self.p_dim*(self.p_dim-1)/2, tf.int32) # number of theta in the model


    def createModelVariables_hs(self, batch_size = 1):
        #
        #
        # rule:  self.trainable_vars[0, 2, 4] must be corresponding spline regression parameters for p_dim>1
        # in 1-d case, self.trainable_vars[0] must be ga_delta parameters, no ga_theta included.
        
        # initial values are quite important for training
        p = int(self.y_ft.shape[2])
        size_delta = int(self.Xmat_delta.shape[1])
        size_theta = int(self.Xmat_theta.shape[1])

        #initializer = tf.initializers.GlorotUniform() # xavier initializer
        #initializer = tf.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        #initializer = tf.initializers.zeros()
        
        # better to have deterministic inital on reg coef to control
        ga_initializer = tf.initializers.zeros()
        ga_initializer_para = tf.initializers.constant(value=0.0)
        ga_initializer_para2 = tf.initializers.constant(value=0.0)
        if size_delta <= 10:
            cvec_d = 0.
        else:
            cvec_d = tf.concat([tf.zeros(10-2)+0., tf.zeros(size_delta-10)+1.], 0)
        if size_theta <= 10:
            cvec_o = 0.5
        else:
            cvec_o = tf.concat([tf.zeros(10)+0.5, tf.zeros(size_theta-10)+1.5], 0)
        
        ga_delta = tf.Variable(ga_initializer_para(shape=(batch_size, p, size_delta), dtype = tf.float32), name='ga_delta')
        lla_delta = tf.Variable(ga_initializer(shape=(batch_size, p, size_theta-2), dtype = tf.float32)-cvec_d, name = 'lla_delta')
        ltau = tf.Variable(ga_initializer(shape=(batch_size, p, 1), dtype = tf.float32)-1, name = 'ltau')
        self.trainable_vars.append(ga_delta)
        self.trainable_vars.append(lla_delta)
        
        nn = int(self.n_theta) # number of thetas in the model        
        ga_theta_re = tf.Variable(ga_initializer_para2(shape=(batch_size, nn, size_theta), dtype = tf.float32), name='ga_theta_re')
        ga_theta_im = tf.Variable(ga_initializer_para2(shape=(batch_size, nn, size_theta), dtype = tf.float32), name='ga_theta_im')

        lla_theta_re = tf.Variable(ga_initializer(shape=(batch_size, nn, size_theta), dtype = tf.float32)-cvec_o, name = 'lla_theta_re')
        lla_theta_im = tf.Variable(ga_initializer(shape=(batch_size, nn, size_theta), dtype = tf.float32)-cvec_o, name = 'lla_theta_im')

        ltau_theta = tf.Variable(ga_initializer(shape=(batch_size, nn, 1), dtype = tf.float32)-1.5, name = 'ltau_theta')

        self.trainable_vars.append(ga_theta_re)
        self.trainable_vars.append(lla_theta_re)
        self.trainable_vars.append(ga_theta_im)
        self.trainable_vars.append(lla_theta_im)
        
        self.trainable_vars.append(ltau)
        self.trainable_vars.append(ltau_theta)
            
        # params:          self.trainable_vars (ga_delta, lla_delta, 
        #                                       ga_theta_re, lla_theta_re, 
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau, ltau_theta)      


    def loglik(self, params):  # log-likelihood for mvts p_dim > 1
        # y_re:            self.y_re
        # y_im:            self.y_im
        # Z_:              self.Zar
        # X_:              self.Xmat
        # params:          self.trainable_vars (ga_delta, xxx, 
        #                                       ga_theta_re, xxx, 
        #                                       ga_theta_im, xxx, ...)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters


        xγ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        sum_xγ = - tf.reduce_sum(xγ, [1, 2])
        exp_xγ_inv = tf.exp(- xγ)

        xα = tf.matmul(self.Xmat_theta, tf.transpose(params[2], [0, 2, 1]))  # no need \ here
        xβ = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0, 2, 1]))

        # Z = Sum [(xα + i xβ) * y]
        Z_theta_re = tf.linalg.matvec(tf.expand_dims(self.Z_re, 0), xα) - tf.linalg.matvec(
            tf.expand_dims(self.Z_im, 0), xβ)
        Z_theta_im = tf.linalg.matvec(tf.expand_dims(self.Z_re, 0), xβ) + tf.linalg.matvec(
            tf.expand_dims(self.Z_im, 0), xα)


        u_re = self.y_re - Z_theta_re
        u_im = self.y_im - Z_theta_im

        numerator = tf.square(u_re) + tf.square(u_im)
        internal = tf.multiply(numerator, exp_xγ_inv)
        tmp2_ = - tf.reduce_sum(internal, [-2, -1]) # sum over p_dim and freq
        log_lik = tf.reduce_sum(sum_xγ + tmp2_) # sum over all LnL
        return log_lik                 


    # Sparse form of loglik()
    def loglik_sparse(self, params):
        # y_re:            self.y_re
        # y_im:            self.y_im
        # Z_:              self.Zar
        # X_:              self.Xmat
        # params:          self.trainable_vars (ga_delta, xxx, 
        #                                       ga_theta_re, xxx, 
        #                                       ga_theta_im, xxx, ...)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters
     
        ldelta_ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0,2,1]))
        tmp1_ = - tf.reduce_sum(ldelta_, [1,2])
        #delta_ = tf.exp(ldelta_)
        delta_inv = tf.exp(- ldelta_)
        theta_re = tf.matmul(self.Xmat_theta, tf.transpose(params[2], [0,2,1]))  # no need \ here
        theta_im = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0,2,1]))
    
        Z_theta_re_ls = [tf.sparse.sparse_dense_matmul(self.Z_re[i], tf.transpose(theta_re[:,i])) - tf.sparse.sparse_dense_matmul(self.Z_im[i], tf.transpose(theta_im[:,i])) for i in range(self.num_obs)]
        Z_theta_im_ls = [tf.sparse.sparse_dense_matmul(self.Z_re[i], tf.transpose(theta_im[:,i])) + tf.sparse.sparse_dense_matmul(self.Z_im[i], tf.transpose(theta_re[:,i])) for i in range(self.num_obs)]

        u_re = self.y_re - tf.transpose(tf.stack(Z_theta_re_ls), [2,0,1])
        u_im = self.y_im - tf.transpose(tf.stack(Z_theta_im_ls), [2,0,1])

        tmp2_ = - tf.reduce_sum(tf.multiply(tf.square(u_re) + tf.square(u_im), delta_inv), [1,2])
        log_lik = tmp1_ + tmp2_
        return log_lik                          
       
#
# Model training one step            
#
    def train_one_step(self, optimizer, loglik, prior): # one step training
        with tf.GradientTape() as tape:
            loss = - loglik(self.trainable_vars) - prior(self.trainable_vars)  # negative log posterior           
        grads = tape.gradient(loss, self.trainable_vars)
        optimizer.apply_gradients(zip(grads, self.trainable_vars))
        return - loss # return log posterior
        

# For new prior strategy, need new createModelVariables() and logprior()
 
    def logprior_hs(self, params):
        # hyper:           list of hyperparameters (tau0, c2, sig2_alp, degree_fluctuate)
        # params:          self.trainable_vars (ga_delta, lla_delta, 
        #                                       ga_theta_re, lla_theta_re, 
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau, ltau_theta)      
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters        
        Sigma1 = tf.multiply(tf.eye(tf.constant(2), dtype=tf.float32), self.hyper[2])
        priorDist1 = tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(Sigma1)) # can also use tfd.MultivariateNormalDiag
        
        Sigm = tfb.Sigmoid()
        s_la_alp = Sigm(- tf.range(1, params[1].shape[-1] + 1., dtype=tf.float32) + self.hyper[3])
        priorDist_la_alp = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_alp)
        
        s_la_theta = Sigm(- tf.range(1, params[3].shape[-1] + 1., dtype=tf.float32) + self.hyper[3])
        priorDist_la_theta = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_theta)

        a2 = tf.square(tf.exp(params[1])) 
        Sigma2i_diag = tf.divide(tf.multiply(tf.multiply(a2, tf.square(tf.exp(params[6]))) , self.hyper[1]),
                          tf.multiply(a2, tf.square(tf.exp(params[6]))) + self.hyper[1] )
            
        priorDist2 = tfd.MultivariateNormalDiag(scale_diag = Sigma2i_diag)
            
        lpriorAlp_delt = tf.reduce_sum(priorDist1.log_prob(params[0][:, :, 0:2]), [1]) #
        lprior_delt = tf.reduce_sum(priorDist2.log_prob(params[0][:, :, 2:]), [1]) # only 2 dim due to log_prob rm the event_shape dim
        lpriorla_delt = tf.reduce_sum(priorDist_la_alp.log_prob(tf.exp(params[1])), [1,2]) + tf.reduce_sum(params[1],[1,2])
        lpriorDel = lprior_delt + lpriorla_delt + lpriorAlp_delt
        
        
        a3 = tf.square(tf.exp(params[3]))
        Sigma3i_diag = tf.divide(tf.multiply(tf.multiply(a3, tf.square(tf.exp(params[7]))) , self.hyper[1]),
                           tf.multiply(a3, tf.square(tf.exp(params[7]))) + self.hyper[1] )
            
        priorDist3 = tfd.MultivariateNormalDiag(scale_diag = Sigma3i_diag)
            
        lprior_thet_re = tf.reduce_sum(priorDist3.log_prob(params[2]), [1])
        lpriorla_thet_re = tf.reduce_sum(priorDist_la_theta.log_prob(tf.exp(params[3])), [1,2]) + tf.reduce_sum(params[3],[1,2])
        lpriorThe_re = lprior_thet_re + lpriorla_thet_re
        
        
        a4 = tf.square(tf.exp(params[5]))
        Sigma4i_diag = tf.divide(tf.multiply(tf.multiply(a4, tf.square(tf.exp(params[7]))) , self.hyper[1]),
                          tf.multiply(a4, tf.square(tf.exp(params[7]))) + self.hyper[1] )
            
        priorDist4 = tfd.MultivariateNormalDiag(scale_diag = Sigma4i_diag)
            
        lprior_thet_im = tf.reduce_sum(priorDist4.log_prob(params[4]),[1])
        lpriorla_thet_im = tf.reduce_sum(priorDist_la_theta.log_prob(tf.exp(params[5])), [1,2]) + tf.reduce_sum(params[5],[1,2])
        lpriorThe_im = lprior_thet_im + lpriorla_thet_im 
        
        
        priorDist_tau = tfd.HalfCauchy(tf.constant(0, tf.float32), self.hyper[0])
        logPrior = lpriorDel + lpriorThe_re + lpriorThe_im + tf.reduce_sum(priorDist_tau.log_prob(tf.exp(params[6])) + params[6], [1,2]) + tf.reduce_sum(priorDist_tau.log_prob(tf.exp(params[7]))+params[7], [1, 2]) 
        return logPrior

    
class SpecVI:
    def __init__(self, x):
        self.data = x
    
    def runModel(self, N_delta=30, N_theta=30, lr_map=5e-4, ntrain_map=5e3, inference_size=500, 
                 nchunks = 400, variation_factor=0, sparse_op=False):
        self.sparse_op = sparse_op
        
        x = self.data
        print('data shape: '+ str(x.shape))

        ## Hyperparameter
        ##
        hyper_hs = []
        tau0=0.01
        c2 = 4
        sig2_alp = 10
        degree_fluctuate = N_delta # the smaller tends to be smoother
        hyper_hs.extend([tau0, c2, sig2_alp, degree_fluctuate])
        
        ## Define Model
        ##
        Spec_hs = SpecModel(x, hyper_hs, sparse_op=self.sparse_op, nchunks=nchunks)
        self.model = Spec_hs # save model object
        # comput fft
        Spec_hs.sc_fft()
        # compute array of design matrix Z, 3d
        if self.sparse_op == False:
            Spec_hs.Zmtrix()
        else:
            Spec_hs.SparseZmtrix()
        # compute X matrix related to basis function on ffreq
        Spec_hs.Xmtrix(N_delta, N_theta)
        # convert all above to tensorflow object
        Spec_hs.toTensor()
        # create tranable variables
        Spec_hs.createModelVariables_hs()
        
        print('Start Model Inference Training: ')
        
        '''
        # Phase1 obtain MAP
        '''
        lr = lr_map
        n_train = ntrain_map #
        optimizer_hs = tf.keras.optimizers.Adam(lr)  

        start_total = timeit.default_timer()
        start_map = timeit.default_timer()
        # train
        @tf.function
        def train_hs(model, optimizer, n_train):
            # model:    model object
            # optimizer
            # n_train:  times of training
            n_samp = model.trainable_vars[0].shape[0]
            lpost = tf.constant(0.0, tf.float32, [n_samp])
            lp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            
            for i in tf.range(n_train):
                if self.sparse_op == False:
                    lpost = model.train_one_step(optimizer, model.loglik, model.logprior_hs)
                else:
                    lpost = model.train_one_step(optimizer, model.loglik_sparse, model.logprior_hs)
                if optimizer.iterations % 500 == 0:
                    tf.print('Step', optimizer.iterations, ': log posterior', lpost)
                lp = lp.write(tf.cast(i, tf.int32), lpost)
            return model.trainable_vars, lp.stack()
        
        print('Start Point Estimating: ')
        opt_vars_hs, lp_hs = train_hs(Spec_hs, optimizer_hs, n_train)    
        # opt_vars_hs:         self.trainable_vars(ga_delta, lla_delta, 
        #                                       ga_theta_re, lla_theta_re, 
        #                                       ga_theta_im, lla_theta_im, 
        #                                       ltau)
        # Variational inference for regression parameters
        end_map = timeit.default_timer()
        print('MAP Training Time: ', end_map - start_map)  
        self.lp = lp_hs
        
        
        '''
        Phase 2 UQ
        '''
        optimizer_vi = tf.optimizers.Adam(5e-2)
        if variation_factor <= 0:
            trainable_Mvnormal = tfd.JointDistributionSequential([
                tfd.Independent(
                tfd.MultivariateNormalDiag(loc = opt_vars_hs[i][0], 
                                           scale_diag = tfp.util.TransformedVariable(tf.constant(1e-4, tf.float32, opt_vars_hs[i][0].shape), tfb.Softplus(), name='q_z_scale')) , 
                reinterpreted_batch_ndims=1)
                for i in tf.range(len(opt_vars_hs))])
        else: # variation_factor > 0
            trainable_Mvnormal = tfd.JointDistributionSequential([
                tfd.Independent(
                tfd.MultivariateNormalDiagPlusLowRank(loc = opt_vars_hs[i][0], 
                                           scale_diag = tfp.util.TransformedVariable(tf.constant(1e-4, tf.float32, opt_vars_hs[i][0].shape), tfb.Softplus()), 
                                           scale_perturb_factor=tfp.util.TransformedVariable(tf.random_uniform_initializer()(opt_vars_hs[i][0].shape + variation_factor), tfb.Identity())), 
                reinterpreted_batch_ndims=1)
                for i in tf.range(len(opt_vars_hs))])
    
                    
        if self.sparse_op == False:
            def conditioned_log_prob(*z):
                return Spec_hs.loglik(z) + Spec_hs.logprior_hs(z)
        else:
            def conditioned_log_prob(*z):
                return Spec_hs.loglik_sparse(z) + Spec_hs.logprior_hs(z)
        
        print('Start UQ training: ')
        start = timeit.default_timer()
        losses = tf.function(lambda l: tfp.vi.fit_surrogate_posterior(target_log_prob_fn=l, surrogate_posterior=trainable_Mvnormal, optimizer=optimizer_vi, num_steps=500*2))(conditioned_log_prob) #
        stop = timeit.default_timer()
        print('VI Time: ', stop - start)  
        stop_total = timeit.default_timer()  
        self.kld = losses
        
        print('Total Inference Training Time: ', stop_total - start_total)  
        
        self.posteriorPointEst = trainable_Mvnormal.mean()
        self.posteriorPointEstStd = trainable_Mvnormal.stddev()
        self.variationalDistribution = trainable_Mvnormal
        

        samp = trainable_Mvnormal.sample(inference_size)
        Spec_hs.freq = Spec_hs.sc_fft()["fq_y"]
        Xmat_delta, Xmat_theta = Spec_hs.Xmtrix(N_delta=N_delta, N_theta=N_theta)
        Spec_hs.toTensor()
        
        
        return losses, Spec_hs, samp
    
    
    
        
        

