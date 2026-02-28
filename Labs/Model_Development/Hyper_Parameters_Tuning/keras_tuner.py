# Ungraded Lab: Intro to Keras Tuner
# Converted from Keras_Tuner.ipynb
# Run: python keras_tuner.py

from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt

tensorBoard_callback = keras.callbacks.TensorBoard("./tb_logs")

# --- Download and prepare the dataset (Fashion MNIST) ---
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
img_train = img_train.astype("float32") / 255.0
img_test = img_test.astype("float32") / 255.0

# --- Baseline model (fixed hyperparameters) ---
b_model = keras.Sequential()
b_model.add(keras.layers.Flatten(input_shape=(28, 28)))
b_model.add(keras.layers.Dense(units=512, activation="relu", name="dense_1"))
b_model.add(keras.layers.Dropout(0.2))
b_model.add(keras.layers.Dense(10, activation="softmax"))
b_model.summary()

b_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

NUM_EPOCHS = 15
b_model.fit(
    img_train,
    label_train,
    epochs=NUM_EPOCHS,
    validation_split=0.2,
    callbacks=[tensorBoard_callback],
)

b_eval_dict = b_model.evaluate(img_test, label_test, return_dict=True)


def print_results(model, model_name, layer_name, eval_dict):
    """Print hyperparameters and evaluation results."""
    print(f"\n{model_name}:")
    print(f"number of units in 1st Dense layer: {model.get_layer(layer_name).units}")
    print(f"learning rate for the optimizer: {model.optimizer.learning_rate.numpy()}")
    for key, value in eval_dict.items():
        print(f"{key}: {value}")


print_results(b_model, "BASELINE MODEL", "dense_1", b_eval_dict)


# --- Hypermodel: model builder for Keras Tuner ---
def model_builder(hp):
    """Build model with tunable hyperparameters (units, units_2, activation, dropout, learning_rate)."""
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    hp_activation = hp.Choice("activation", values=["relu", "tanh"])
    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation, name="tuned_dense_1"))

    hp_units_2 = hp.Int("units_2", min_value=32, max_value=256, step=32)
    model.add(keras.layers.Dense(units=hp_units_2, activation="relu", name="tuned_dense_2"))

    hp_dropout = hp.Choice("dropout", values=[0.1, 0.2, 0.3, 0.4])
    model.add(keras.layers.Dropout(hp_dropout))
    model.add(keras.layers.Dense(10, activation="softmax"))

    hp_learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# --- Instantiate RandomSearch tuner (optimize val_loss) ---
tuner = kt.RandomSearch(
    model_builder,
    objective=kt.Objective("val_loss", direction="min"),
    max_trials=20,
    executions_per_trial=1,
    directory="kt_dir",
    project_name="kt_random",
)

tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
tensorboard_tuner_callback = keras.callbacks.TensorBoard(log_dir="./keras_tuner", update_freq="batch")

# Perform hypertuning
tuner.search(
    img_train,
    label_train,
    epochs=NUM_EPOCHS,
    validation_split=0.2,
    callbacks=[stop_early, tensorboard_tuner_callback],
)

# --- Best hyperparameters and final model ---
best_hps = tuner.get_best_hyperparameters()[0]
print(f"""
The hyperparameter search is complete.
Optimal units (1st Dense): {best_hps.get('units')}
Optimal units (2nd Dense): {best_hps.get('units_2')}
Optimal activation: {best_hps.get('activation')}
Optimal dropout: {best_hps.get('dropout')}
Optimal learning rate: {best_hps.get('learning_rate')}
""")

h_model = tuner.hypermodel.build(best_hps)
h_model.summary()

h_model.fit(img_train, label_train, epochs=NUM_EPOCHS, validation_split=0.2)
h_eval_dict = h_model.evaluate(img_test, label_test, return_dict=True)

print_results(b_model, "BASELINE MODEL", "dense_1", b_eval_dict)
print_results(h_model, "HYPERTUNED MODEL", "tuned_dense_1", h_eval_dict)
