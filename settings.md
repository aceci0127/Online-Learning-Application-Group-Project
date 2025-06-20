# Setting
    A company has to choose prices dynamically.

        Parameters
            Number of rounds T
            Number of types of products N
            Set of possible prices P (small and discrete set)
            Production capacity B ▲ ! For simplicity, there is a total number of products B that the company can produce (independently from the specific type of product)

        Buyer behavior
            Has a valuation vi for each type of product in N
            Buys all products priced below their respective valuations


# Interaction
    At each round t ∈ T :
        1 The company chooses which types of product to sell and set price pi for each type of product
        2 A buyer with a valuation for each type of product arrives
        3 The buyer buys a unit of each product with price smaller than the product valuation