import random

class BTreeNode:
    """
    A node in the B-tree structure.
    
    Attributes:
        leaf (bool): Indicates if the node is a leaf node
        t (int): Minimum degree of the B-tree
        keys (list): List of customer IDs stored in the node
        values (list): List of customer records corresponding to the keys
        children (list): List of child nodes
    """
    def __init__(self, leaf=True, t=3):
        self.leaf = leaf
        self.t = t
        self.keys = []    # Initialize as separate empty lists
        self.values = []  # Instead of pointing to the same list
        self.children = []

class CustomerRecord:
    """
    Stores customer information for the microfinance system.
    
    Attributes:
        customer_id (int): Unique identifier for the customer
        name (str): Customer's full name
        location (str): Customer's geographical location
        credit_score (int): Customer's credit score
        loans (list): List of LoanRecord objects associated with the customer
    """
    def __init__(self, customer_id, name, location, credit_score):
        self.customer_id = customer_id
        self.name = name
        self.location = location
        self.credit_score = credit_score
        self.loans = []

class LoanRecord:
    """
    Stores loan information for individual customer transactions.
    
    Attributes:
        loan_id (int): Unique identifier for the loan
        amount (float): Principal amount of the loan
        interest_rate (float): Annual interest rate as a decimal
        term_months (int): Loan duration in months
        status (str): Current status of the loan (e.g., 'Active', 'Pending')
    """
    def __init__(self, loan_id, amount, interest_rate, term_months, status):
        self.loan_id = loan_id
        self.amount = amount
        self.interest_rate = interest_rate
        self.term_months = term_months
        self.status = status

class MicrofinanceBTree:
    """
    B-tree implementation for managing microfinance customer records.
    """
    def __init__(self, t=3):
        self.root = BTreeNode(t=t)
        self.t = t

    def insert(self, customer_record):
        """
        Inserts a new customer record into the B-tree.
        """
        root = self.root
        if len(root.keys) == (2 * self.t - 1):
            self.root = new_root = BTreeNode(leaf=False, t=self.t)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, customer_record)
        else:
            self._insert_non_full(root, customer_record)

    def _split_child(self, parent, index):
        """
        Splits a child node when it reaches maximum capacity.
        """
        t = self.t
        child = parent.children[index]
        new_child = BTreeNode(leaf=child.leaf, t=t)
        
        parent.keys.insert(index, child.keys[t-1])
        parent.values.insert(index, child.values[t-1])
        
        new_child.keys = child.keys[t:]
        new_child.values = child.values[t:]
        child.keys = child.keys[:t-1]
        child.values = child.values[:t-1]
        
        if not child.leaf:
            new_child.children = child.children[t:]
            child.children = child.children[:t]
        
        parent.children.insert(index + 1, new_child)

    def _insert_non_full(self, node, record):
        """
        Inserts a record into a non-full node.
        """
        i = len(node.keys) - 1
        if node.leaf:
            while i >= 0 and record.customer_id < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, record.customer_id)  # Store customer_id in keys
            node.values.insert(i + 1, record)
        else:
            while i >= 0 and record.customer_id < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.t - 1):
                self._split_child(node, i)
                if record.customer_id > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], record)

    def search(self, customer_id):
        """
        Searches for a customer record by ID.
        """
        return self._search_recursive(self.root, customer_id)

    def _search_recursive(self, node, customer_id):
        """
        Recursively searches for a customer record in the B-tree.
        """
        i = 0
        while i < len(node.keys) and customer_id > node.keys[i]:
            i += 1
        if i < len(node.keys) and customer_id == node.keys[i]:
            return node.values[i]
        return None if node.leaf else self._search_recursive(node.children[i], customer_id)

def create_dummy_data(loan_recs: int = 3, total_customers: int = 3):
    """
    Creates a B-tree with sample customer and loan data for testing.
    """
    btree = MicrofinanceBTree(t=3)
    
   
    loans = [
        LoanRecord(id, amount, rate, months, "Active")
        for id, amount, rate, months in [
            (indx, random.randint(500, 5000), random.uniform(0, 1), random.randint(1, 48)) for indx in range(1, loan_recs) # 48 months = 4 years
        ]
    ]
    
    # dummy names and locations banks.
    locations = ['North Legon', 'East Legon', 'Madina', 'Accra', 'Kasoa', 'Ashiama', 'Lapas', 'Shiashi']
    names = ['Akwesi', 'Stigar', 'Bertha', 'Gifty', 'Prince', 'Arthur', 'Ofori', 'Linda', 'Samuel', 'Simon', 'Osei']

    # create customer records using nested list comprehensions
    customers = [
        CustomerRecord(id, name, location, score)
        for id, name, location, score in [
            (100 + indx, f"{random.choice(names)} {random.choice(names)}", random.choice(locations), random.randint(500, 1000)) for indx in range(1, total_customers)
           
        ]
    ]
    
    # assign loans to customers.
    for customer, loan in zip(customers[:total_customers], loans):
        customer.loans.append(loan)
    
    for customer in customers:
        btree.insert(customer)
    
    return btree

def test_btree(total_loans: int, total_customers: int):
    """
    Tests the B-tree implementation with sample data.
    """
    btree = create_dummy_data(loan_recs=total_loans, total_customers=total_customers)
    
    test_ids = [101, 103, 108, 105, 111, 201, 999]
    
    for customer_id in test_ids:
        print(f"\nSearching for customer ID {customer_id}:")
        result = btree.search(customer_id)
        
        if result:
            print(f"Found: {result.name} from {result.location}")
            print(f"Number of loans: {len(result.loans)}")
            [print(f"Loan amount: ${loan.amount}, Interest rate: {loan.interest_rate*100}%") 
             for loan in result.loans]
        else:
            print("Customer not found")

if __name__ == "__main__":
    customers = input('Enter Total Customers: ')
    loans = input('Enter Total Loans: ')
    test_btree(int(loans), int(customers))

