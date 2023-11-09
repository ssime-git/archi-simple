import pickle 

with open('./decision_tree_classifier.pkl', 'wb') as f:
    pickle.dump("test.pickle", f)

try:
    with open("test.pickle", "rb") as f:
        print(f.read())
        model = pickle.load(f)

except FileNotFoundError as e:
    print(e)
