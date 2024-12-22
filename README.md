# One-Shot-Image-Recognition
Implementation of the paper "Siamese Neural Networks for One-shot Image Recognition". 
You can find the paper here "https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf"


- In this paper, the authors present a novel approach which limits assumptions on the structure of the inputs while automatically acquiring features which enable the model to generalize successfully from few examples.

An example from the paper is given below:
<img width="633" alt="image" src="https://github.com/user-attachments/assets/7d440d27-4156-4e7d-91a1-333a49bcb8ef">


### WHy Siamese Nets
Large Siamese Convolutional Nets are employed here because
a) are capable of learning generic image features useful for making predictions about unknown class distributions even when very few examples from these new distributions are available
b) are easily trained using standard optimization techniques on pairs sampled from the source data and 
c) provide a competitive approach that does not rely upon domain-specific knowledge by instead exploiting deep learning techniques.

(Note: Feature here means a transformation of data that is useful for learning)

### Architecture

<img width="521" alt="image" src="https://github.com/user-attachments/assets/67e5a298-245c-4ecd-8eea-de761c9a8b06" />

x -> Conv Layers -> 

### Model

 - The stride is fixed to be 1 always.
 - Loss Function: A regularized cross-entropy function is used. <img width="564" alt="image" src="https://github.com/user-attachments/assets/a5fe52ba-0ac5-4930-9b3c-c69b33b89182" />
 - Rectified linear (ReLU) units in the first L − 2 layers and sigmoidal units in the remaining layers.
 - The max pooling layer's filter size and stride is 2.
 - The weighted L1 distance is used between the twin feature vectors h1 and h2 combined with a sigmoid activation, which maps onto the interval [0, 1].
 - All network weights in the convolutional layers from a normal distribution with zero-mean and a standard deviation of 0.01.
 - Biases were also initialized from a normal distribution, but with mean 0.5 and standard deviation 0.01.
 - In the fully connected layer, the weights were drawn from a much wider normal distribution with zero-mean and a standard deviation of of 0.02, but the biases were intialized in the same way.
 - Learning rates are decayed uniformly across the network by 1 percent per epoch.


<img width="1265" alt="image" src="https://github.com/user-attachments/assets/feb5fe62-6a38-420d-90f0-be7bdf0eb167" />
   
### Dataset
<img width="533" alt="image" src="https://github.com/user-attachments/assets/b7ebc373-6c6b-4c2a-a7a9-760be5bc9223" />
The Omniglot dataset is a collection of 1623 hand drawn characters from 50 alphabets. For every character there are just 20 examples, each drawn by a different person at resolution 105x105.

The number of letters in each alphabet varies considerably from about 15 to upwards of 40 characters. All characters across these alphabets are produced a single time by each of 20 drawers Lake split the data into a 40 alphabet background set and a 10 alphabet evaluation set.

(Note: Here the background set and training set are not the same, refer to the paper section 4.1 for more information)

### Affline Distortion
In the training set, the images are augmented using small affline distortions and the new transformations were determined stochastically by a multidimensional uniform distribution (every component of the trasformation is included with a probability of 0.5).




(All information presented here is inspired from the above mentioned paper)


