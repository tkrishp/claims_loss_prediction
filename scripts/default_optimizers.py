from keras.optimizers import SGD, Adagrad, Adadelta, Adamax

def_sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
def_adagrad = Adagrad(lr=0.01, epsilon=1e-06)
def_adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
def_adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)