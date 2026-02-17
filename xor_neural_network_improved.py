

import random
import math

random.seed(10)


def matrix_mul(a, b):
    """Multiply two 2D matrices.
    
    Args:
        a: Matrix of shape (m, n)
        b: Matrix of shape (n, p)
    
    Returns:
        Matrix of shape (m, p)
    """
    c = [[0 for i in range(len(b[0]))] for j in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]
    return c


class Layer:
    """Neural network layer with weight updates via backpropagation.
    
    Attributes:
        num_inputs: Number of input features
        num_neurons: Number of neurons in the layer
        weights: Weight matrix (num_inputs x num_neurons)
        biases: Bias vector (num_neurons)
        activation: Activation function name ('relu' or 'sigmoid')
        lr: Learning rate for weight updates
    """
    
    def __init__(self, num_inputs, num_neurons, activation='relu', lr=0.001):
        """Initialize layer with Xavier-initialized weights and zero biases.
        
        Args:
            num_inputs: Number of input features
            num_neurons: Number of neurons
            activation: Activation function ('relu' or 'sigmoid')
            lr: Learning rate (default: 0.001)
        
        Raises:
            ValueError: If num_inputs or num_neurons are non-positive
        """
        if num_inputs <= 0 or num_neurons <= 0:
            raise ValueError("num_inputs and num_neurons must be positive")
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        
        # Xavier/Glorot initialization for weights (improves training stability)
        # Draws from uniform distribution [-limit, limit]
        xavier_limit = math.sqrt(6.0 / (num_inputs + num_neurons))
        self.weights = [
            [random.uniform(-xavier_limit, xavier_limit) for _ in range(num_neurons)]
            for _ in range(num_inputs)
        ]
        
        # Initialize biases to 0 (more standard than 1)
        self.biases = [0.0 for _ in range(num_neurons)]
        
        # Set activation function and its derivative
        activation_functions = {
            "relu": (self.relu, self.relu_der),
            "sigmoid": (self.sigmoid, self.sigmoid_der)
        }
        
        if activation not in activation_functions:
            raise ValueError(f"activation must be 'relu' or 'sigmoid', got '{activation}'")
        
        self.act_func = activation_functions[activation][0]
        self.act_der = activation_functions[activation][1]
        
        # Cache for forward/backward pass
        self.forward_pass = [0.0 for _ in range(num_neurons)]          # Layer output
        self.z = [0.0 for _ in range(num_neurons)]                     # Pre-activation
        self.weight_gradients = [[0.0 for _ in range(num_neurons)] for _ in range(num_inputs)]
        self.bias_gradients = [0.0 for _ in range(num_neurons)]
        self.inputs = [0.0 for _ in range(num_inputs)]                 # Previous layer output
        
        self.lr = lr  # Learning rate
    
    def __str__(self):
        """Return string representation of layer weights and biases."""
        s = "Weights:\n"
        for i in range(self.num_inputs):
            s += "  " + "  ".join(f"{w:7.4f}" for w in self.weights[i]) + "\n"
        s += "Biases: " + "  ".join(f"{b:7.4f}" for b in self.biases) + "\n"
        return s
    
    def forward(self, inputs):
        """Forward pass through the layer.
        
        Args:
            inputs: Input matrix of shape (batch_size, num_inputs)
        
        Returns:
            Output activations of shape (batch_size, num_neurons)
        """
        self.inputs = inputs[0]
        res = matrix_mul(inputs, self.weights)
        
        for i in range(self.num_neurons):
            # z = w^T * x + b (pre-activation)
            self.z[i] = res[0][i] + self.biases[i]
            # Apply activation function
            self.forward_pass[i] = self.act_func(self.z[i])
            res[0][i] = self.forward_pass[i]
        
        return res
    
    def backward(self, dL_by_dNouts):
        """Backward pass (backpropagation) through the layer.
        
        Computes gradients for weights and biases, updates parameters,
        and returns gradients for the previous layer.
        
        Args:
            dL_by_dNouts: Gradient of loss w.r.t. layer outputs (dL/dOut)
        
        Returns:
            Gradients to pass to previous layer (dL/dIn)
        """
        error_signals = [0.0 for _ in range(self.num_neurons)]
        
        # Step 1: Calculate error signals using chain rule
        # error = dL/dOut * dOut/dZ = dL/dOut * activation'(z)
        for i in range(self.num_neurons):
            error_signals[i] = dL_by_dNouts[i] * self.act_der(self.z[i])
        
        # Step 2: Calculate weight gradients
        # dL/dW[i][j] = error[j] * input[i]
        for i in range(self.num_inputs):
            for j in range(self.num_neurons):
                dL_by_dWij = error_signals[j] * self.inputs[i]
                self.weight_gradients[i][j] = dL_by_dWij
        
        # Bias gradients are just the error signals
        self.bias_gradients = error_signals
        
        # Step 3: Compute gradients for previous layer (backpropagate errors)
        # dL/dInput[i] = sum_j(error[j] * W[i][j])
        prev_layer_dl_by_dnouts = [0.0 for _ in range(self.num_inputs)]
        for i in range(self.num_inputs):
            for j in range(self.num_neurons):
                prev_layer_dl_by_dnouts[i] += error_signals[j] * self.weights[i][j]
        
        # Step 4: Update weights and biases using gradient descent
        # W := W - lr * dL/dW
        for i in range(self.num_inputs):
            for j in range(self.num_neurons):
                self.weights[i][j] -= self.lr * self.weight_gradients[i][j]
        
        # B := B - lr * dL/dB
        for i in range(self.num_neurons):
            self.biases[i] -= self.lr * self.bias_gradients[i]
        
        return prev_layer_dl_by_dnouts
    
    def relu(self, x):
        """ReLU (Rectified Linear Unit) activation function."""
        return max(0.0, x)
    
    def relu_der(self, x):
        """Derivative of ReLU activation function."""
        return 0.0 if x <= 0.0 else 1.0
    
    def sigmoid(self, x):
        """Sigmoid activation function: 1 / (1 + e^-x)"""
        return 1.0 / (1.0 + math.exp(-x))
    
    def sigmoid_der(self, z):
        """Derivative of sigmoid activation function: sigma(z) * (1 - sigma(z))"""
        sig = self.sigmoid(z)
        return sig * (1.0 - sig)


# ============================================================================
# Text Preprocessing and Vectorization
# ============================================================================

def create_vocabulary(sentences):
    """Create a vocabulary from a list of sentences.
    
    Args:
        sentences: List of sentence strings
    
    Returns:
        Dictionary mapping words to indices
    """
    vocab = {}
    words = set()
    
    for sentence in sentences:
        # Split and lowercase
        tokens = sentence.lower().split()
        words.update(tokens)
    
    # Create vocab with index for each unique word
    for idx, word in enumerate(sorted(words)):
        vocab[word] = idx
    
    return vocab


def sentence_to_vector(sentence, vocab, vector_size=None):
    """Convert a sentence to a numerical feature vector using word frequency.
    
    Args:
        sentence: Input sentence string
        vocab: Vocabulary dictionary from create_vocabulary()
        vector_size: Size of output vector (default: len(vocab))
    
    Returns:
        List of floats representing the sentence
    """
    if vector_size is None:
        vector_size = len(vocab)
    
    vector = [0.0] * vector_size
    tokens = sentence.lower().split()
    
    # Word frequency encoding
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            if idx < vector_size:
                vector[idx] += 1.0
    
    # Normalize by dividing by number of words (prevents length bias)
    if len(tokens) > 0:
        vector = [v / len(tokens) for v in vector]
    
    return vector


def one_hot_encode(label, num_classes=4):
    """Convert a class label to one-hot encoding.
    
    Args:
        label: Class index (0, 1, 2, or 3)
        num_classes: Total number of classes
    
    Returns:
        List with one-hot encoding (e.g., [1, 0, 0, 0] for class 0)
    """
    vector = [0.0] * num_classes
    vector[label] = 1.0
    return vector


# ============================================================================
# Loss Functions
# ============================================================================

def cross_entropy_loss(y_actual, y_predicted):
    """Calculate cross-entropy loss for classification.
    
    Args:
        y_actual: One-hot encoded target vector
        y_predicted: Softmax probability output
    
    Returns:
        Cross-entropy loss (negative log likelihood)
    """
    epsilon = 1e-10  # Avoid log(0)
    loss = 0.0
    for i in range(len(y_actual)):
        if y_actual[i] > 0:  # Only sum for true class
            loss -= y_actual[i] * math.log(max(epsilon, y_predicted[i]))
    return loss


def mean_cross_entropy_loss(y_actuals, y_predicteds):
    """Calculate mean cross-entropy loss over all samples.
    
    Args:
        y_actuals: List of one-hot encoded targets
        y_predicteds: List of softmax probability vectors
    
    Returns:
        Mean cross-entropy loss
    """
    if len(y_actuals) == 0:
        return 0.0
    
    total_loss = 0.0
    for y_actual, y_pred in zip(y_actuals, y_predicteds):
        total_loss += cross_entropy_loss(y_actual, y_pred)
    
    return total_loss / len(y_actuals)


def softmax(logits):
    """Apply softmax function to convert logits to probabilities.
    
    Args:
        logits: List of raw output values from network
    
    Returns:
        List of probabilities that sum to 1.0
    """
    # Subtract max for numerical stability
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    sum_exp = sum(exp_logits)
    return [exp_logit / sum_exp for exp_logit in exp_logits]


def squared_loss(y_actual, y_predicted):
    """Calculate squared error loss for a single sample.
    
    Args:
        y_actual: Actual target value
        y_predicted: Model prediction
    
    Returns:
        Squared error: (y_actual - y_predicted)^2
    """
    return (y_actual - y_predicted) ** 2


def mean_squared_loss(y_actual, y_predicted):
    """Calculate mean squared error loss over all samples.
    
    Args:
        y_actual: List of actual values
        y_predicted: List of predicted values
    
    Returns:
        Mean squared error across all samples
    """
    if len(y_actual) == 0:
        return 0.0
    
    total_loss = 0.0
    for i in range(len(y_actual)):
        total_loss += squared_loss(y_actual[i], y_predicted[i])
    
    return total_loss / len(y_actual)


# ============================================================================
# Multi-Head Attention Layer
# ============================================================================

class AttentionHead:
    """Single attention head for multi-head attention mechanism.
    
    Performs scaled dot-product attention: softmax(Q*K^T / sqrt(d_k)) * V
    """
    
    def __init__(self, input_dim, head_dim, lr=0.01):
        """Initialize attention head projections.
        
        Args:
            input_dim: Input feature dimension
            head_dim: Output dimension per head
            lr: Learning rate
        """
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.lr = lr
        
        # Query, Key, Value projection matrices
        xavier_limit = math.sqrt(6.0 / (input_dim + head_dim))
        self.W_q = [[random.uniform(-xavier_limit, xavier_limit) for _ in range(head_dim)] 
                    for _ in range(input_dim)]
        self.W_k = [[random.uniform(-xavier_limit, xavier_limit) for _ in range(head_dim)] 
                    for _ in range(input_dim)]
        self.W_v = [[random.uniform(-xavier_limit, xavier_limit) for _ in range(head_dim)] 
                    for _ in range(input_dim)]
        
        # Gradients
        self.dW_q = [[0.0 for _ in range(head_dim)] for _ in range(input_dim)]
        self.dW_k = [[0.0 for _ in range(head_dim)] for _ in range(input_dim)]
        self.dW_v = [[0.0 for _ in range(head_dim)] for _ in range(input_dim)]
        
        # Cache for backward pass
        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None
    
    def forward(self, x):
        """Forward pass for this attention head.
        
        Args:
            x: Input vector of shape (input_dim,)
        
        Returns:
            Output of shape (head_dim,)
        """
        # Project to Q, K, V
        self.Q = [sum(x[i] * self.W_q[i][j] for i in range(self.input_dim)) 
                  for j in range(self.head_dim)]
        self.K = [sum(x[i] * self.W_k[i][j] for i in range(self.input_dim)) 
                  for j in range(self.head_dim)]
        self.V = [sum(x[i] * self.W_v[i][j] for i in range(self.input_dim)) 
                  for j in range(self.head_dim)]
        
        # Scaled dot-product attention: softmax(Q*K^T / sqrt(d_k))
        scale = math.sqrt(self.head_dim)
        scores = sum(self.Q[i] * self.K[i] for i in range(self.head_dim)) / scale
        
        # Apply softmax (simplified for single attention score)
        self.attention_weights = 1.0 / (1.0 + math.exp(-scores))  # Sigmoid as approximation
        
        # Apply attention to values
        output = [self.attention_weights * v for v in self.V]
        
        return output
    
    def backward(self, grad_output, x):
        """Backward pass for this attention head.
        
        Args:
            grad_output: Gradient w.r.t. output (head_dim,)
            x: Original input (input_dim,)
        
        Returns:
            Gradient w.r.t. input (input_dim,)
        """
        # Gradient w.r.t. attention weights
        grad_attention = sum(grad_output[i] * self.V[i] for i in range(self.head_dim))
        grad_attention *= self.attention_weights * (1 - self.attention_weights)  # Sigmoid derivative
        
        # Gradient w.r.t. Q, K, V
        scale = math.sqrt(self.head_dim)
        grad_q = [grad_attention * self.K[i] / scale for i in range(self.head_dim)]
        grad_k = [grad_attention * self.Q[i] / scale for i in range(self.head_dim)]
        grad_v = [self.attention_weights * grad_output[i] for i in range(self.head_dim)]
        
        # Gradient w.r.t. W_q, W_k, W_v
        for i in range(self.input_dim):
            for j in range(self.head_dim):
                self.dW_q[i][j] = grad_q[j] * x[i]
                self.dW_k[i][j] = grad_k[j] * x[i]
                self.dW_v[i][j] = grad_v[j] * x[i]
                
                # Update weights
                self.W_q[i][j] -= self.lr * self.dW_q[i][j]
                self.W_k[i][j] -= self.lr * self.dW_k[i][j]
                self.W_v[i][j] -= self.lr * self.dW_v[i][j]
        
        # Gradient w.r.t. input
        grad_x = [0.0] * self.input_dim
        for i in range(self.input_dim):
            for j in range(self.head_dim):
                grad_x[i] += grad_q[j] * self.W_q[i][j]
                grad_x[i] += grad_k[j] * self.W_k[i][j]
                grad_x[i] += grad_v[j] * self.W_v[i][j]
        
        return grad_x


class MultiHeadAttention:
    """Multi-head attention layer with 12 attention heads.
    
    Runs multiple attention heads in parallel and concatenates outputs.
    """
    
    def __init__(self, input_dim, num_heads=12, lr=0.01):
        """Initialize multi-head attention.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads (default: 12)
            lr: Learning rate
        """
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})")
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.lr = lr
        
        # Create 12 attention heads
        self.heads = [AttentionHead(input_dim, self.head_dim, lr) for _ in range(num_heads)]
        
        # Output projection
        xavier_limit = math.sqrt(6.0 / (input_dim + input_dim))
        self.W_out = [[random.uniform(-xavier_limit, xavier_limit) for _ in range(input_dim)] 
                      for _ in range(input_dim)]
        self.b_out = [0.0] * input_dim
        self.dW_out = [[0.0 for _ in range(input_dim)] for _ in range(input_dim)]
        self.db_out = [0.0] * input_dim
        
        self.output_before_projection = None
    
    def forward(self, x):
        """Forward pass for multi-head attention.
        
        Args:
            x: Input vector of shape (input_dim,)
        
        Returns:
            Output of shape (input_dim,)
        """
        # Run all 12 heads in parallel
        head_outputs = [head.forward(x) for head in self.heads]
        
        # Concatenate all head outputs
        concatenated = []
        for head_output in head_outputs:
            concatenated.extend(head_output)
        
        self.output_before_projection = concatenated
        
        # Final linear projection
        output = [0.0] * self.input_dim
        for i in range(self.input_dim):
            output[i] = self.b_out[i]
            for j in range(self.input_dim):
                output[i] += concatenated[j] * self.W_out[j][i]
        
        return output
    
    def backward(self, grad_output, x):
        """Backward pass for multi-head attention.
        
        Args:
            grad_output: Gradient w.r.t. output (input_dim,)
            x: Original input (input_dim,)
        
        Returns:
            Gradient w.r.t. input (input_dim,)
        """
        # Gradient w.r.t. output projection
        grad_concat = [0.0] * self.input_dim
        for i in range(self.input_dim):
            for j in range(self.input_dim):
                self.dW_out[j][i] = grad_output[i] * self.output_before_projection[j]
                grad_concat[j] += grad_output[i] * self.W_out[j][i]
                
                # Update output weights
                self.W_out[j][i] -= self.lr * self.dW_out[j][i]
            
            self.db_out[i] = grad_output[i]
            self.b_out[i] -= self.lr * self.db_out[i]
        
        # Propagate gradients back to each head
        grad_x_total = [0.0] * self.input_dim
        start_idx = 0
        
        for head_idx, head in enumerate(self.heads):
            end_idx = start_idx + self.head_dim
            head_grad = grad_concat[start_idx:end_idx]
            
            # Backward pass through each head
            grad_x = head.backward(head_grad, x)
            for i in range(self.input_dim):
                grad_x_total[i] += grad_x[i]
            
            start_idx = end_idx
        
        return grad_x_total


# ============================================================================
# Problem Classification Neural Network
# ============================================================================

class ProblemClassifier:
    """Multi-class classifier for problem type detection.
    
    Classifies sentences into categories:
    - 0: Mechanical problem
    - 1: Electrical problem
    - 2: HVAC problem
    - 3: Software problem
    """
    
    CLASSES = {
        0: "Mechanical",
        1: "Electrical",
        2: "HVAC",
        3: "Software"
    }
    
    def __init__(self, vocab_size, hidden_size=16, lr=0.01):
        """Initialize the classifier.
        
        Args:
            vocab_size: Size of vocabulary/input dimension
            hidden_size: Number of neurons in hidden layer
            lr: Learning rate
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocabulary = {}
        
        # Create layers: input -> hidden -> output(4 classes)
        self.hidden_layer = Layer(vocab_size, hidden_size, activation="sigmoid", lr=lr)
        self.output_layer = Layer(hidden_size, 4, activation="sigmoid", lr=lr)
    
    def forward(self, feature_vector):
        """Forward pass through the network.
        
        Args:
            feature_vector: Numerical representation of sentence
        
        Returns:
            Softmax probabilities for each class
        """
        hidden_out = self.hidden_layer.forward([feature_vector])
        raw_output = self.output_layer.forward(hidden_out)
        probabilities = softmax(raw_output[0])
        return probabilities
    
    def backward(self, loss_gradient):
        """Backward pass through the network.
        
        Args:
            loss_gradient: Gradient of loss w.r.t. output
        """
        hidden_gradient = self.output_layer.backward(loss_gradient)
        self.hidden_layer.backward(hidden_gradient)
    
    def train_on_batch(self, feature_vector, target_one_hot):
        """Train on a single sample.
        
        Args:
            feature_vector: Numerical representation of sentence
            target_one_hot: One-hot encoded target label
        
        Returns:
            Cross-entropy loss
        """
        # Forward pass
        predictions = self.forward(feature_vector)
        
        # Calculate loss and its gradient
        loss = cross_entropy_loss(target_one_hot, predictions)
        
        # Gradient of cross-entropy loss w.r.t. output
        # dL/dOutput[i] = -target[i] / pred[i] (simplified)
        loss_gradient = [
            predictions[i] - target_one_hot[i] for i in range(4)
        ]
        
        # Backward pass
        self.backward(loss_gradient)
        
        return loss
    
    def predict(self, sentence):
        """Predict the problem class for a sentence.
        
        Args:
            sentence: Input sentence string
        
        Returns:
            Tuple of (predicted_class_index, predicted_class_name, confidence_scores)
        """
        feature_vector = sentence_to_vector(sentence, self.vocabulary)
        probabilities = self.forward(feature_vector)
        
        # Get prediction
        predicted_class = probabilities.index(max(probabilities))
        class_name = self.CLASSES[predicted_class]
        confidence = probabilities[predicted_class]
        
        return predicted_class, class_name, confidence, probabilities


def create_problem_classifier(sentences, hidden_size=16, lr=0.01):
    """Factory function to create a classifier with vocabulary from training data.
    
    Args:
        sentences: List of training sentences
        hidden_size: Number of hidden neurons
        lr: Learning rate
    
    Returns:
        Initialized ProblemClassifier instance
    """
    vocab = create_vocabulary(sentences)
    classifier = ProblemClassifier(len(vocab), hidden_size, lr)
    classifier.vocabulary = vocab
    return classifier


# ============================================================================
# Training
# ============================================================================

if __name__ == "__main__":
    # Training data: sentences and their problem types
    # 0 = Mechanical, 1 = Electrical, 2 = HVAC, 3 = Software
    training_data = [
        # Mechanical problems (15 samples)
        ("The motor is making a grinding noise during operation", 0),
        ("Bearing wear detected in the pump system", 0),
        ("Transmission hydraulic fluid is leaking from the seal", 0),
        ("Gear teeth are damaged and causing vibration", 0),
        ("The conveyor belt keeps slipping on the pulley", 0),
        ("Mechanical fasteners are loose and assembly is failing", 0),
        ("Shaft alignment is off causing excessive vibration", 0),
        ("Coupling failed between motor and pump", 0),
        ("Bearing seizure detected in rotational equipment", 0),
        ("Hydraulic pressure exceeds safe limits", 0),
        ("Chain tension is too loose causing skip", 0),
        ("Mechanical seal is worn out and leaking", 0),
        ("Ball joint deterioration affecting equipment stability", 0),
        ("Tension in drive belt has deteriorated significantly", 0),
        ("Mechanical resistance increased due to corrosion", 0),
        
        # Electrical problems (15 samples)
        ("The circuit breaker trips every time we power on", 1),
        ("Electrical wiring is overheating causing voltage drop", 1),
        ("Transformer is showing signs of electrical failure", 1),
        ("Capacitor failed in the power supply unit", 1),
        ("Motor windings are short circuiting under load", 1),
        ("Electrical grounding is insufficient for safety", 1),
        ("Battery voltage is unstable causing system shutdown", 1),
        ("Relay coil is not responding to input signals", 1),
        ("Fuse keeps blowing in the main distribution board", 1),
        ("Electrical insulation is compromised in the cable", 1),
        ("Phase imbalance detected in three-phase system", 1),
        ("Voltage regulation module is faulty", 1),
        ("Electrical continuity test failed", 1),
        ("Surge protection device needs replacement", 1),
        ("Power factor is critically low", 1),
        
        # HVAC problems (15 samples)
        ("Air conditioning system is not cooling the room", 2),
        ("Heating unit failed during winter operation", 2),
        ("HVAC thermostat is not responding to temperature changes", 2),
        ("Refrigerant leak detected in the AC system", 2),
        ("Air circulation fan is broken in the ventilation unit", 2),
        ("HVAC compressor is making unusual noise", 2),
        ("Ductwork is blocked preventing proper air flow", 2),
        ("Evaporator coil is frozen restricting flow", 2),
        ("HVAC blower motor bearing is failing", 2),
        ("Condenser is clogged with debris", 2),
        ("Expansion valve malfunction reducing cooling efficiency", 2),
        ("Fan blade is bent causing vibration", 2),
        ("HVAC ductwork has multiple insulation breaches", 2),
        ("Temperature sensor is reading incorrect values", 2),
        ("Humidifier in HVAC system is not working", 2),
        
        # Software problems (15 samples)
        ("Application crashes when processing large files", 3),
        ("Database connection timeout errors", 3),
        ("Software cannot load configuration from file", 3),
        ("Memory leak detected in long running process", 3),
        ("API endpoint returning HTTP 500 error", 3),
        ("User authentication system is completely broken", 3),
        ("Software license validation is failing", 3),
        ("UI freezes when performing search operation", 3),
        ("Data synchronization between servers is failing", 3),
        ("Software update installation fails repeatedly", 3),
        ("Error log shows critical exception in third party library", 3),
        ("Performance degradation after recent software patch", 3),
        ("Software unable to connect to required service", 3),
        ("Code compilation fails with syntax errors", 3),
        ("Software displays corrupted data on screen", 3),
    ]
    
    # Extract sentences and labels
    training_sentences = [item[0] for item in training_data]
    training_labels = [item[1] for item in training_data]
    
    # Create classifier
    print("=" * 70)
    print("Problem Type Classification Neural Network")
    print("=" * 70)
    print(f"Classes: Mechanical (0), Electrical (1), HVAC (2), Software (3)")
    print(f"Training samples: {len(training_sentences)}")
    
    classifier = create_problem_classifier(training_sentences, hidden_size=16, lr=0.05)
    vocab_size = len(classifier.vocabulary)
    print(f"Vocabulary size: {vocab_size} words\n")
    
    # Training loop
    epochs = 8000
    log_interval = 800
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for sentence, label in training_data:
            feature_vector = sentence_to_vector(sentence, classifier.vocabulary)
            target_one_hot = one_hot_encode(label)
            loss = classifier.train_on_batch(feature_vector, target_one_hot)
            epoch_loss += loss
        
        # Log progress
        if epoch % log_interval == 0:
            avg_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.6f}")
    
    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}\n")
    
    # Testing the classifier on new sentences
    test_sentences = [
        "The bearing is making noise when the shaft rotates",
        "Electrical short circuit in the main panel",
        "AC unit is not cooling properly in summer",
        "Hydraulic pump seal is leaking fluid everywhere",
        "Circuit board components are burning out",
        "Thermostat is not controlling HVAC temperature",
        "Application keeps crashing when saving files",
        "Software update failed with critical error",
        "Motor coupling alignment needs adjustment",
        "Power supply voltage fluctuation issue",
    ]
    

    for test_sentence in test_sentences:
        pred_class, class_name, confidence, probs = classifier.predict(test_sentence)
        print(f"\nSentence: {test_sentence}")
        print(f"Predicted: {class_name} (confidence: {confidence:.2%})")
        print(f"Probabilities - Mechanical: {probs[0]:.2%}, Electrical: {probs[1]:.2%}, HVAC: {probs[2]:.2%}, Software: {probs[3]:.2%}")
    
    print(f"\n{'=' * 70}")
