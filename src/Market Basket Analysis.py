# Code based on:
# https://mahendra-choudhary.medium.com/learning-market-basket-analysis-using-association-rule-mining-in-python-d0aa2c0ea9a9
# https://medium.com/analytics-vidhya/association-analysis-in-python-2b955d0180c
# https://www.kaggle.com/code/mervetorkan/association-rules-with-python

import csv

def load_transactions(file_name):
    """Loads transactions from a CSV file and returns a list of lists."""
    transactions = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            transactions.append(row[0].split(','))
    return transactions

def calculate_support(transactions, item_set):
    """
    Calculates the support of an item or item set in transactions.

    Support is the proportion of transactions that contain the item set.
    """
    count = 0
    for transaction in transactions:
        if all(item in transaction for item in item_set):
            count += 1
    return count / len(transactions)

def calculate_confidence(transactions, item_set_A, item_set_B):
    """
    Calculates the confidence of item_set_A leading to item_set_B in transactions.

    Confidence is the proportion of transactions containing item_set_A that also contain item_set_B.
    """
    count_A_B = 0
    count_A = 0
    for transaction in transactions:
        if all(item in transaction for item in item_set_A):
            count_A += 1
            if all(item in transaction for item in item_set_B):
                count_A_B += 1
    if count_A == 0:
        return 0
    return count_A_B / count_A

def main():
    transactions = load_transactions('shoppingtransactions.csv')
    while True:
        print("\nMenu:")
        print("1. sup item[,item]")
        print("2. con item[,item] --> item[,item]")
        print("3. exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            items = input("Enter item(s) separated by commas: ").split(',')
            support = calculate_support(transactions, items)
            print("Support of {}: {:.2f}".format(','.join(items), support))
        
        elif choice == '2':
            input_items = input("Enter item(s) to calculate confidence: ").split('-->')
            item_A = input_items[0].split(',')
            item_B = input_items[1].split(',')
            confidence = calculate_confidence(transactions, item_A, item_B)
            print("Confidence of {} --> {}: {:.2f}".format(','.join(item_A), ','.join(item_B), confidence))
        
        elif choice == '3':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

