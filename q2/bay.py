import pandas as pd

class BayesianBeliefNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_name, values):
        self.nodes[node_name] = values
        self.edges[node_name] = []

    def add_edge(self, parent, child):
        self.edges[parent].append(child)

    def fiti(self, data):
        # Learn conditional probability tables (CPTs)
        for node in self.nodes:
            parents = self.edges[node]
            if not parents:
                # If no parents, calculate the marginal probabilities
                self.nodes[node] = self.calculate_marginal(data, node)
            else:
                # If there are parents, calculate conditional probabilities
                self.nodes[node] = self.calculate_conditional(data, node, parents)

    def calculate_conditional(self, data, node, parents):
        conditional_data = data.groupby(parents + [node]).size().unstack(fill_value=0)
        conditional_probabilities = conditional_data.div(conditional_data.sum(axis=1), axis=0)
        return conditional_probabilities.to_dict()

    def predict(self, evidence):
        # Perform inference
        probabilities = {}
        for node in self.nodes:
            if node not in evidence:
                # If the node itself is not in evidence, use the unique values to calculate probabilities
                unique_values = self.nodes[node]
                probabilities[node] = {value: self.calculate_probability(node, evidence, value) for value in unique_values}
        return probabilities

    def calculate_marginal(self, data, node):
        counts = data[node].value_counts()
        probabilities = counts / len(data)
        return probabilities.to_dict()

    def get_parent_values(self, parents, evidence):
         return [evidence[parent] for parent in parents]

    def calculate_probability(self, node, evidence, value):
    # Calculate P(node = value | evidence)
        parents = self.edges[node]
        if not parents:
        # If no parents, return the marginal probability
            return self.nodes[node][value]
        else:
        # If there are parents, return the conditional probability
            parent_values = self.get_parent_values(parents, evidence)
            return self.nodes[node][tuple(parent_values)][value]


# Example usage with the given dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

from pgmpy.models import BayesianNetwork
model = BayesianNetwork([('Outlook', 'PlayTennis'), ('Temperature', 'PlayTennis'),
                       ('Humidity', 'PlayTennis'), ('Wind', 'PlayTennis')])

from pgmpy.estimators import MaximumLikelihoodEstimator
model.fit(data, estimator=MaximumLikelihoodEstimator)

bbn = BayesianBeliefNetwork()
bbn.add_node('Outlook', data['Outlook'].unique())
bbn.add_node('Temperature', data['Temperature'].unique())
bbn.add_node('Humidity', data['Humidity'].unique())
bbn.add_node('Wind', data['Wind'].unique())
bbn.add_node('PlayTennis', data['PlayTennis'].unique())

bbn.add_edge('Outlook', 'PlayTennis')
bbn.add_edge('Temperature', 'PlayTennis')
bbn.add_edge('Humidity', 'PlayTennis')
bbn.add_edge('Wind', 'PlayTennis')

bbn.fiti(data)

from pgmpy.inference import VariableElimination
inference = VariableElimination(model)

# Take the evidence from the user
outlook = input("Enter Outlook (Rain/Sunny/Overcast): ")
temperature = input("Enter Temperature (Hot/Mild/Cool): ")
humidity = input("Enter Humidity (High/Normal): ")
wind = input("Enter Wind (Weak/Strong): ")

query_result = inference.query(variables=['PlayTennis'], evidence={'Outlook': outlook, 'Temperature': temperature, 'Humidity': humidity, 'Wind': wind})

df = pd.DataFrame({query_result.variables[0]: list(query_result.state_names[query_result.variables[0]]), 
                   "Probability": query_result.values})

print(df)