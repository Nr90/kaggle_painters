
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
import numpy as np
import disk_util
from imagefromcsvgenerator import ImageFromCSVGenerator

seed = 1
nb_painters = 1584
batch = 16
train_csv = 'data/train.csv'
test_csv = 'data/test.csv'
validation_csv = 'data/validation.csv'
step1optimizer = 'adam'
step2optimizer = SGD(lr=0.0001, momentum=0.9)
metrics = ['accuracy']
checkpoint_file = 'ckpt.h5'
output_model = 'painters_model_tensorflow.h5'
log_dir = './log/'
metrics = ['accuracy']

np.random.seed(seed)

train_samples = disk_util.countlines(train_csv) / batch * batch
print 'Train samples ' + str(train_samples)
test_samples = disk_util.countlines(test_csv) / batch * batch
print 'Test samples ' + str(test_samples)
validation_samples = disk_util.countlines(validation_csv) / batch * batch
print 'Validation samples ' + str(validation_samples)

callbacks = [
	ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_best_only=True),
	TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True),
	EarlyStopping(monitor='val_loss', patience=3, verbose=1)
]

csvgenerator = ImageFromCSVGenerator()
train_generator = csvgenerator.flow_from_csv(train_csv, batch)
test_generator = csvgenerator.flow_from_csv(test_csv, batch)
validation_generator = csvgenerator.flow_from_csv(validation_csv, batch)
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_painters, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
	layer.trainable = False

model.compile(
	optimizer=step1optimizer,
	metrics=metrics,
	loss='categorical_crossentropy'
)

model.fit_generator(
	train_generator,
	samples_per_epoch=train_samples,
	nb_epoch=100,
	validation_data=test_generator,
	nb_val_samples=test_samples,
	callbacks=callbacks
)

# Restore best checkpoint
model.load_weights(checkpoint_file)

for layer in base_model.layers:
	layer.trainable = True

model.compile(
	optimizer=step2optimizer,
	loss='categorical_crossentropy',
	metrics=metrics
)

model.fit_generator(
	train_generator,
	samples_per_epoch=train_samples,
	nb_epoch=100,
	validation_data=test_generator,
	nb_val_samples=test_samples,
	callbacks=callbacks
)
model.load_weights(checkpoint_file)
model.save(output_model)

# Evaluate model
print 'Starting evaluation'
scores = model.evaluate_generator(
	validation_generator,
	val_samples=validation_samples
)

print("%s: %.4f" % (model.metrics_names[0], scores[0]))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
