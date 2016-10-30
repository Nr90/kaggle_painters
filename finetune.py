from neon.backends import gen_backend
from neon.transforms import Rectlin
from neon.initializers import Constant, Xavier
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.initializers import GlorotUniform
from neon.transforms import Softmax
from neon.models import Model
from neon.data.datasets import Dataset
from neon.util.persist import load_obj
from neon.data import ImageLoader
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.transforms import CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.transforms import Misclassification
import os


painters = 1584
be = gen_backend(batch_size=16)
data_dir = 'data/macrobatch_out'


relu = Rectlin()
conv_params = {
	'strides': 1,
	'padding': 1,
	'init': Xavier(local=True),
	'bias': Constant(0),
	'activation': relu}

# Set up the model layers
vgg_layers = []

# set up 3x3 conv stacks with different number of filters
vgg_layers.append(Conv((3, 3, 64), **conv_params))
vgg_layers.append(Conv((3, 3, 64), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 128), **conv_params))
vgg_layers.append(Conv((3, 3, 128), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(
	Affine(nout=4096, init=GlorotUniform(), bias=Constant(0), activation=relu))
vgg_layers.append(Dropout(keep=0.5))
vgg_layers.append(
	Affine(nout=4096, init=GlorotUniform(), bias=Constant(0), activation=relu))
vgg_layers.append(Dropout(keep=0.5))
vgg_layers.append(Affine(
	nout=painters, init=GlorotUniform(), bias=Constant(0),
	activation=Softmax(), name="class_layer"))
model = Model(layers=vgg_layers)

# location and size of the VGG weights file
url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
filename = 'VGG_D.p'
size = 554227541

# edit filepath below if you have the file elsewhere
_, filepath = Dataset._valid_path_append('data', '', filename)
if not os.path.exists(filepath):
	Dataset.fetch_dataset(url, filename, filepath, size)

# load the weights param file
print("Loading VGG weights from {}...".format(filepath))
trained_vgg = load_obj(filepath)
print("Done!")

param_layers = [l for l in model.layers.layers]
param_dict_list = trained_vgg['model']['config']['layers']
for layer, params in zip(param_layers, param_dict_list):
	if(layer.name == 'class_layer'):
		break

	# To be sure, we print the name of the layer in our model
	# and the name in the vgg model.
	print(layer.name + ", " + params['config']['name'])
	layer.load_weights(params, load_states=True)

img_set_options = dict(repo_dir=data_dir, inner_size=224)
train = ImageLoader(
	set_name='train', scale_range=(256, 384),
	shuffle=True, **img_set_options)
test = ImageLoader(
	set_name='validation', scale_range=(256, 384),
	shuffle=True, **img_set_options)

# define different optimizers for the class_layer and the rest of the network
# we use a momentum coefficient of 0.9 and weight decay of 0.0005.
opt_vgg = GradientDescentMomentum(0.001, 0.9, wdecay=0.0005)
opt_class_layer = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005)

# also define optimizers for the bias layers, which have a different learning
# rate and not weight decay.
opt_bias = GradientDescentMomentum(0.002, 0.9)
opt_bias_class = GradientDescentMomentum(0.02, 0.9)

# set up the mapping of layers to optimizers
opt = MultiOptimizer({
	'default': opt_vgg, 'Bias': opt_bias,
	'class_layer': opt_class_layer, 'class_layer_bias': opt_bias_class
})

# use cross-entropy cost to train the network
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

valmetric = Misclassification()

callbacks = Callbacks(model, eval_set=test, metric=valmetric)
model.fit(train, optimizer=opt, num_epochs=10, cost=cost, callbacks=callbacks)
