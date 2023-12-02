This Python script defines a class `BayesianBeliefNetwork` to create, fit, and predict with a Bayesian Belief Network (BBN). Additionally, an example is provided using the pgmpy library for Bayesian network modeling.

Here's a breakdown of the code:

1. **BayesianBeliefNetwork Class:**
   - The class is initialized with empty dictionaries `nodes` and `edges` to store information about the nodes and edges in the Bayesian network.

2. **add_node Method:**
   - Adds a node to the Bayesian network along with its possible values.

3. **add_edge Method:**
   - Adds a directed edge between parent and child nodes in the Bayesian network.

4. **fiti Method:**
   - Learns the conditional probability tables (CPTs) from the provided data for each node in the network.

5. **calculate_conditional Method:**
   - Calculates conditional probabilities based on the provided data, given a node and its parents.

6. **predict Method:**
   - Performs inference to predict probabilities for unobserved nodes given evidence.

7. **calculate_marginal Method:**
   - Calculates marginal probabilities for a node based on the provided data.

8. **get_parent_values Method:**
   - Retrieves values of parent nodes from evidence.

9. **calculate_probability Method:**
   - Calculates the probability of a node taking a specific value given the evidence.

10. **Example Usage:**
   - A sample dataset related to playing tennis is created using pandas DataFrame.
   - A Bayesian network model is defined using the pgmpy library.
   - An instance of the `BayesianBeliefNetwork` class is created, nodes and edges are added, and the model is fitted with the provided data.
   - VariableElimination from pgmpy is used for Bayesian inference based on the provided evidence (Outlook, Temperature, Humidity, Wind).

The script allows users to enter evidence (Outlook, Temperature, Humidity, Wind) and predicts the probability of playing tennis. The results are displayed in a DataFrame.

The pgmpy library is used for defining the Bayesian network structure and learning parameters from data, while the custom `BayesianBeliefNetwork` class is employed for simplicity in learning and predicting based on the Bayesian network.