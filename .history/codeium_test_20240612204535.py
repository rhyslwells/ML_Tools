def helloworld():
    print("Hello World")

import pandas as pd
import numpy as np


helloworld()

def create_person_weight_table(person_names, person_weights):
    """
    Creates a table showing person names and their respective weights.

    Parameters:
    person_names (list): A list of strings representing the names of the people.
    person_weights (list): A list of integers representing the weights of the people.

    Returns:
    DataFrame: A table with columns 'Person' and 'Weight (kg)'.
    """
    data = {'Person': person_names, 'Weight (kg)': person_weights}
    return pd.DataFrame(data)


# Example usage
person_names = ['Alice', 'Bob', 'Charlie']
person_weights = [70, 80, 90]
table = create_person_weight_table(person_names, person_weights)
print(table)


