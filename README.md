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

* - need to try 0.01
 
|Task ID|Score in original paper|My score|
|---|---|---|
|   1|   |  100 |
|   2|   |  83.8 |
|   3|   |   |*
|   4|   |100|
|   5|   |98.7|


|   6|   |98.1|*
|   7|   |92.9| 
|   8|   |   error|
|   9|   |100|
|  10|  |99.8|*
|  11|   | 99.6|*
|  12|   |   100|
|  13|   |   99.8|*
|  14|   |   82.1|*
|  15|   |   99.9|*
|  16|   |  44.7 |*
|  17|   |   61.4|*
|  18|   |  89.4 |
|  19|   |  error |
|  20|   |  100 |

For the future:Try and make DMN work with real-world text, like Wikipedia data. Find a use case for this architecture.
