# City SSFP

A small project that fetches real-time data from streets in Chicago city and utilizes graph algorithms and a custom Neural Network to predict traffic and find the single source fastest path from one segment to another

---

## How it works

The main idea is construct a graph from the retrived data and run a "dynamic dijkstra" algorithm that updates the graph edges using a neural network to predict traffic and calculate the fastest route.

The neural network uses LSTM layers that allows to predict the next timestamp for each segment given a certain amount of previous data.

While running dijkstra, the neural network is tasked to predict the future traffic speed, after that, the output is combined with the street length and velocity to calculate the expected time for each adjacent edges at the current timestamp.

---

## Checklist

- [x] Graph and SSSP
- [X] Retrieve SSFP
- [x] Neural Network
- [X] Multiple Timestamps Prediction 
- [X] Dynamic Dijkstra
- [X] Neural Network Chart Visualization
- [ ] Function Filter (Convolutional, Linear, etc)
- [ ] Graph Visualization

--- 

## Pros / Cons

Pros:
1. Much more accurate prediction of time
2. Retrieves the fastest route
3. Helps to monitor traffic in all segments of the city

Cons: 
1. Requires a generous and abundant dataset
2. Slow data fetching
3. Prediction makes SSFP algorithm much slower (compared to usual SSSP)

Which currently makes it not viable for realtime purposes