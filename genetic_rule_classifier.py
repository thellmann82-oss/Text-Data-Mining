import random
import operator
import numpy as np

from deap import base, creator, tools
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Daten
# ----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_features = X.shape[1]

# ----------------------------
# Regel (Baustein)
# ----------------------------
class Rule:
    def __init__(self):
        self.conditions = []
        self.label = random.choice([0, 1])

        for _ in range(random.randint(1, 3)):
            feature = random.randint(0, n_features - 1)
            threshold = random.uniform(X_train[:, feature].min(),
                                       X_train[:, feature].max())
            op = random.choice(["<", ">"])
            self.conditions.append((feature, op, threshold))

    def matches(self, x):
        for f, op, t in self.conditions:
            if op == "<" and not (x[f] < t):
                return False
            if op == ">" and not (x[f] > t):
                return False
        return True

# ----------------------------
# Individuum = Regelmenge
# ----------------------------
class RuleSet:
    def __init__(self):
        self.rules = [Rule() for _ in range(random.randint(1, 5))]

    def predict(self, X):
        preds = []
        for x in X:
            pred = None
            for rule in self.rules:
                if rule.matches(x):
                    pred = rule.label
                    break
            if pred is None:
                pred = 0  # Default-Klasse
            preds.append(pred)
        return np.array(preds)

# ----------------------------
# Fitness
# ----------------------------
def fitness(individual):
    preds = individual.predict(X_train)
    acc = accuracy_score(y_train, preds)

    # Strafe für Komplexität
    complexity = sum(len(r.conditions) for r in individual.rules)
    penalty = 0.01 * complexity

    return acc - penalty

# ----------------------------
# Evolution
# ----------------------------
def mutate(ind):
    new = RuleSet()
    new.rules = ind.rules.copy()

    if random.random() < 0.5 and len(new.rules) > 1:
        new.rules.pop(random.randint(0, len(new.rules)-1))
    else:
        new.rules.append(Rule())

    return new

def crossover(ind1, ind2):
    new = RuleSet()
    cut1 = random.randint(0, len(ind1.rules))
    cut2 = random.randint(0, len(ind2.rules))
    new.rules = ind1.rules[:cut1] + ind2.rules[cut2:]
    return new

# ----------------------------
# Initialisierung
# ----------------------------
population_size = 50
generations = 30

population = [RuleSet() for _ in range(population_size)]

# ----------------------------
# Evolution Loop
# ----------------------------
for gen in range(generations):
    scored = [(ind, fitness(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)

    best, best_score = scored[0]
    print(f"Generation {gen}: Beste Fitness = {best_score:.4f}")

    survivors = [ind for ind, _ in scored[:10]]

    new_population = survivors.copy()

    while len(new_population) < population_size:
        if random.random() < 0.5:
            parent = random.choice(survivors)
            child = mutate(parent)
        else:
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)

        new_population.append(child)

    population = new_population

# ----------------------------
# Test
# ----------------------------
best_model = best
test_preds = best_model.predict(X_test)

print("\nTestgenauigkeit:", accuracy_score(y_test, test_preds))

print("\nGefundene Regeln:")
for i, rule in enumerate(best_model.rules):
    print(f"Regel {i+1}:")
    for cond in rule.conditions:
        f, op, t = cond
        print(f"  Merkmal[{f}] {op} {t:.3f}")
    print(f"  DANN Klasse = {rule.label}")
