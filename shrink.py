import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model('ai_human_detector_v2.h5')

# Convert to TFLite with Float16 Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16] # Shrinks size by 50-75%

tflite_model = converter.convert()

# Save the quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)