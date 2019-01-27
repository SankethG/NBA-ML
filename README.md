# NBA-ML
Simple machine learning algorithm that produces a graph to compare the difference in points between Damian Lillard and Kyrie Irving

TF.PY
the basics of creating a graph using tensorflow and matplotlib. It explains the 4 main steps of making a graph
1. Reset the graph, which is needed as it clears everything in the graph
2. make placeholders/variables, which are going to be used in the graph
3. Do operations on the variables, which will be needed to determine the slope and changes in the graph
4. use Tf.Session(), which runs those operations on the graph

SCATTER.PY
A more complex approach to building a graph, using more sophisticated, complex methods of ML such as:
  Training- inputs a series of values and the graph starts to learn and trains itself to build a graph by minimizing the cost,
  with the use of functions such as the sigmoid function or the RELU function(sigmoid output is value between 0-1, relu either gives a 0 or 1, and is used more nowadays)
  Gradient descent- a math term that is usually presented in multivariable calculus, where the function goes toward the negative gradient
 to find the minimum of the function, in order to optimize an algorithm(In tensorflow, it is called GradientDescentOptimizer)
 
 NBA1.PY
 Combines/adds on to these two graphs by personalizing it. Here, I compared the point difference between Damian Lillard and Kyrie Irving by
 building a graph that uses stats from NBA.COM to predict the graph of the players points per game given 7 values. 




















  
  
