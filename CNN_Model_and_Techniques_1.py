import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
tf.compat.v1.enable_eager_execution()
import shap


# Step 1: Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the data for the CNN
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Define a Simple CNN Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Step 3: Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the Model for Future Use
# model.save('mnist_simple_cnn.h5')


# Step 6: Evaluate the Model

# Get the predicted classes for the test dataset
y_pred_probs = model.predict(x_test)  # Predicted probabilities for each class
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class labels

# Flatten the true labels (y_test is one-hot encoded, so take the argmax)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
class_report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
print("Classification Report:\n")
print(class_report)

##################################################################################################################################################
##################################################################################################################################################

# Step 7: Apply Integrated Gradient Technique

# Function to calculate Integrated Gradients
def integrated_gradients(inputs, model, target_class_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros(inputs.shape)  # Baseline is a black image
    
    # Scale inputs from baseline to the actual input
    scaled_inputs = np.array([baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)])
    
    # Initialize the gradients list
    gradients_list = []
    for scaled_input in scaled_inputs:
        scaled_input = tf.convert_to_tensor(scaled_input.reshape((1, 28, 28, 1)), dtype=tf.float32)  # Convert to tf.Tensor
        with tf.GradientTape() as tape:
            tape.watch(scaled_input)
            preds = model(scaled_input)  # Pass through the model
            loss = preds[:, target_class_idx]  # Target class prediction
        grads = tape.gradient(loss, scaled_input)  # Compute gradients
        gradients_list.append(grads.numpy())
    
    # Compute the average gradients and the integrated gradients
    avg_gradients = np.mean(gradients_list, axis=0)
    integrated_grads = (inputs - baseline) * avg_gradients
    return integrated_grads

# Example: Apply Integrated Gradients for the first test image
img = x_test[0:1]  # First test image
target_class = np.argmax(y_test[0])  # True class for the first test image

# Call the function
ig = integrated_gradients(img, model, target_class)

# # Visualize the Integrated Gradients
# plt.imshow(ig[0, :, :, 0], cmap='viridis')
# plt.colorbar()
# plt.title(f"Integrated Gradients for Class {target_class}")
# plt.show()

# Display additional comparison images
# Function to visualize the original image, IG heatmap, and overlayed IG
def plot_comparison_images(original_image, baseline_image, integrated_grads, target_class):
    # Plot Original Image
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot Perturbed (Baseline) Image
    plt.subplot(2, 2, 2)
    plt.imshow(baseline_image.squeeze(), cmap='gray')
    plt.title('Perturbed (Baseline) Image')
    plt.axis('off')

    # Plot Integrated Gradients heatmap
    plt.subplot(2, 2, 3)
    plt.imshow(integrated_grads.squeeze(), cmap='viridis')
    plt.colorbar()
    plt.title(f"Integrated Gradients for Class {target_class}")
    plt.axis('off')

    # Overlay IG on Original Image
    plt.subplot(2, 2, 4)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.imshow(integrated_grads.squeeze(), cmap='viridis', alpha=0.5)
    plt.title(f"Overlay of IG on Original Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example: Apply Integrated Gradients for the first test image
img = x_test[0:1]  # First test image
baseline_image = np.zeros_like(img)  # Baseline image (black)
target_class = np.argmax(y_test[0])  # True class for the first test image

# Call the function to calculate Integrated Gradients
ig = integrated_gradients(img, model, target_class)

# Call the function to plot the comparison images
print("\n\n\n \033[1mINTEGRATED GRADIENT\033[0m \n")
plot_comparison_images(img, baseline_image, ig, target_class)


##################################################################################################################################################
##################################################################################################################################################

# Step 8: Apply SmmoothGrad Technique

# Function to compute gradients
def compute_gradients(img, model, target_class):
    """
    Computes the gradients of the target class output with respect to the input image.
    
    Parameters:
        img: Input image (batch of 1 image).
        model: Trained model.
        target_class: Target class index.
    
    Returns:
        Gradients of the target class with respect to the input image.
    """
    # Convert the input image to a tf.Tensor if it's a NumPy array
    if isinstance(img, np.ndarray):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Ensure the input image is being tracked by the gradient tape
        tape.watch(img)
        # Get the model's prediction for the target class
        preds = model(img)
        target_output = preds[:, target_class]  # Select the output for the target class
    
    # Compute the gradients of the target class output w.r.t. the input image
    gradients = tape.gradient(target_output, img)
    return gradients.numpy()  # Convert gradients back to a NumPy array

# Function to calculate SmoothGrad
def compute_smoothgrad(img, model, target_class, num_samples=50, noise_level=0.2):
    """
    Computes SmoothGrad by adding noise to the input image and averaging the gradients.
    
    Parameters:
        img: Input image.
        model: Trained model.
        target_class: Target class for which gradients are computed.
        num_samples: Number of noisy samples.
        noise_level: Standard deviation of the Gaussian noise.
    
    Returns:
        SmoothGrad heatmap (averaged gradients).
    """
    # Initialize SmoothGrad array with the same shape as the input image
    smoothgrad = np.zeros_like(img, dtype=np.float32)
    
    for _ in range(num_samples):
        # Add Gaussian noise to the input image
        noisy_img = img + noise_level * np.random.normal(size=img.shape)
        # Clip the noisy image to keep values in the valid range [0, 1]
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # Compute gradients for the noisy image
        grads = compute_gradients(noisy_img, model, target_class)
        
        # Ensure grads has the same shape as the input image
        if grads.shape != img.shape:
            grads = np.mean(grads, axis=-1, keepdims=True)  # Reduce channel dimensions if needed
        
        # Accumulate the gradients into smoothgrad
        smoothgrad += grads
    
    # Average the accumulated gradients
    smoothgrad /= num_samples
    
    return smoothgrad

# Example: Apply SmoothGrad for the first test image
img = x_test[0:1]  # First test image (shape: 1, 28, 28, 1)
target_class = np.argmax(y_test[0])  # True class for the first test image

# Compute SmoothGrad
smoothgrad = compute_smoothgrad(img, model, target_class, num_samples=50, noise_level=0.3)

def plot_smoothgrad_comparison_with_overlay(img, smoothgrad, target_class):
    """
    Visualizes the original image, SmoothGrad heatmap, and an overlay of the SmoothGrad on the image.
    
    Parameters:
        img: Original input image.
        smoothgrad: SmoothGrad heatmap.
        target_class: Target class for the image.
    """
    # Normalize SmoothGrad heatmap to [0, 1]
    smoothgrad_normalized = (smoothgrad - smoothgrad.min()) / (smoothgrad.max() - smoothgrad.min())
    
    # Create an overlay by blending the original image with the SmoothGrad heatmap
    overlay = img[0].squeeze() * 0.5 + smoothgrad_normalized.squeeze() * 0.5  # 50% blend
    
    # Plot the results
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original Image (Class {target_class})")
    plt.imshow(img[0].squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("SmoothGrad Heatmap")
    plt.imshow(smoothgrad_normalized.squeeze(), cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay Image")
    plt.imshow(overlay, cmap='inferno')  # Use an expressive colormap for the overlay
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example: Visualize the overlay
print("\n\n\n \033[1mSmoothGrad\033[0m \n")
plot_smoothgrad_comparison_with_overlay(img, smoothgrad, target_class)

##################################################################################################################################################
##################################################################################################################################################

# Step 9: Hybrid Application of SHAP and Integrated Gradient

# Function to plot SHAP and IG comparisons
def plot_comparison_images(original_image, shap_values, integrated_grads, target_class):
    # Plot SHAP explanation
    shap_sum = np.sum(np.abs(shap_values[0]), axis=-1)  # Summing over the channels for visualization
    
    plt.figure(figsize=(15, 10))

    # Plot Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot SHAP heatmap
    plt.subplot(2, 3, 2)
    plt.imshow(shap_sum, cmap='jet')
    plt.colorbar()
    plt.title(f"SHAP for Class {target_class}")
    plt.axis('off')

    # Plot Integrated Gradients heatmap
    plt.subplot(2, 3, 3)
    plt.imshow(integrated_grads.squeeze(), cmap='viridis')
    plt.colorbar()
    plt.title(f"Integrated Gradients for Class {target_class}")
    plt.axis('off')

    # Overlay SHAP on Original Image
    plt.subplot(2, 3, 4)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.imshow(shap_sum, cmap='jet', alpha=0.5)
    plt.title(f"Overlay of SHAP on Original Image")
    plt.axis('off')

    # Overlay IG on Original Image
    plt.subplot(2, 3, 5)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.imshow(integrated_grads.squeeze(), cmap='viridis', alpha=0.5)
    plt.title(f"Overlay of IG on Original Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Select a test image for explanation
img = x_test[0:1]  # First test image
target_class = np.argmax(y_test[0])  # True class for the first test image

# Step 1: Calculate SHAP values
background = x_train[:100]  # Use 100 examples for SHAP's background data
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(img)

# Step 2: Calculate Integrated Gradients
ig = integrated_gradients(img, model, target_class)

# Step 3: Plot comparison images
print("\n\n\n \033[1mSHAP AND INTEGRATED GRADIENT(IG) HYBRID APPLICATION\033[0m \n")
plot_comparison_images(img, shap_values, ig, target_class)
