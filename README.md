# NeuralNetworkSIMD

Adapted code by Bobby Anguelov. The net is trained to recognize handwritten characters, using the MNIST training data set. A speed up of 17x was attained via improving caching and applying SIMD vectorization.

Due to github file size limitations, the data set is seperate. Please download it from the following link and place it in the data folder, next to 'labels.bin'.
https://drive.google.com/file/d/1BFQVKx9qIKtPGieyo6nh5AnEIwYSxGbV/view?usp=sharing

Please build the program in 64bit
