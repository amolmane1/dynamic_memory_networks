Tensorflow implementation of Dynamic Memory Networks (DMN) (https://arxiv.org/pdf/1603.01417.pdf) 

DMN+ is a state of the art neural network architecture for question answering. It uses memory and attention mechanisms to solve problems like the following:

```
Input:
John went to the bedroom.
John picked up the football.
John went to the garden.

Question:
Where is the football?

Answer:
Garden
```

This implementation uses the Facebook babi dataset and can be trained on 20 different types of tasks like counting, reasoning, induction, deduction, path-finding, etc.

Below are the test accuracies of the model on the 20 tasks:
 
|Task ID|Score in original paper|My score|
|---|---|---|
|   1|   100|  100 |
|   2|   99.7|  83.8 |
|   3|   98.9|56.1|
|   4|   100|100|
|   5|   99.5|98.7|
|   6|   100|100|
|   7|   97.6|92.9| 
|   8|   100|TODO|
|   9|   100|100|
|  10|   100|99.9|
|  11|   100| 100|
|  12|   100|   100|
|  13|   100|   100|
|  14|   99.8|   91.2|
|  15|   100|   100|
|  16|   54.7|  45.8 |
|  17|   95.8|   59.9|
|  18|   97.9|  89.4 |
|  19|   100|  TODO |
|  20|   100|  100 |

For the future: 
- Implement AttnGRU for the Episodic Memory Module
- Try and make DMN work with real-world text, like Wikipedia data. Find a use case for this architecture.
- Implement DMN-powered chatbot
