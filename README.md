# City SSFP

A small project that fetches real-time data from streets in Chicago city and utilizes graph algorithms and a custom Neural Network to predict traffic and find the single source fastest path from one segment to another

---

## How it works

The main idea is construct a graph from the retrived data and run a "dynamic dijkstra" algorithm that updates the graph edges using a neural network to predict traffic and calculate the fastest route.

The neural network uses LSTM layers that allows to predict for `n` timestamps further and returns an array with size `s` (the prediction for each segment).

While running dijkstra, the neural network is tasked to predict the future traffic speed, after that, the output matrix is combined with the street length and velocity to calculate the expected time for each adjacent edges at the current time.