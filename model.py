import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model():
    """
    Crear un modelo CNN para MNIST
    Este modelo utiliza una arquitectura diferente al ejemplo visto en clase
    utilizando capas convolucionales más profundas.
    
    Returns:
        Un modelo compilado de TensorFlow listo para entrenar
    """
    tf.random.set_seed(42)
    
    input_shape = (28, 28, 1)
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(256, activation='relu'),
        Dropout(0.4),  
        
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, filepath):
    """
    Guardar el modelo en el sistema de archivos.
    
    Args:
        model: Modelo de TensorFlow a guardar
        filepath: Ruta donde se guardará el modelo
    """
    model.save(filepath)

def load_model(filepath):
    """
    Cargar un modelo desde el sistema de archivos.
    
    Args:
        filepath: Ruta donde se encuentra el modelo
        
    Returns:
        El modelo cargado
    """
    return tf.keras.models.load_model(filepath)

if __name__ == "__main__":
    model = create_model()
    model.summary()
    
    save_model(model, "modelo_global_inicial.h5")