import random
import numpy as np
from sklearn.metrics import accuracy_score


class Rule:
    def __init__(self, n_features, feature_ranges):
        self.label = random.choice([0, 1])
        self.conditions = []
        for _ in range(random.randint(1, 3)):
            f = random.randint(0, n_features - 1)
            lo, hi = feature_ranges[f]
            t = random.uniform(lo, hi)
            op = random.choice(['<', '>'])
            self.conditions.append((f, op, t))

    def matches(self, x):
        for f, op, t in self.conditions:
            if op == '<' and x[f] >= t:
                return False
            if op == '>' and x[f] <= t:
                return False
        return True

    def to_dict(self, feature_names):
        return {
            'bedingungen': [
                {
                    'merkmal': feature_names[f],
                    'operator': op,
                    'schwellwert': round(float(t), 4),
                }
                for f, op, t in self.conditions
            ],
            'klasse': int(self.label),
            'klasse_text': 'Steigt ↑' if self.label == 1 else 'Fällt ↓',
        }


class RuleSet:
    def __init__(self, rules):
        self.rules = rules

    def predict_one(self, x):
        for rule in self.rules:
            if rule.matches(x):
                return rule.label
        return 0  # Standardklasse: Kurs fällt

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class GeneticRuleClassifier:
    def __init__(self, population_size=50, generations=30,
                 n_survivors=10, complexity_penalty=0.01):
        self.population_size = population_size
        self.generations = generations
        self.n_survivors = n_survivors
        self.complexity_penalty = complexity_penalty
        self.best_model = None
        self.feature_names = None
        self.history = []
        self._n_features = None
        self._feature_ranges = None

    def _new_ruleset(self):
        n = random.randint(1, 5)
        return RuleSet([Rule(self._n_features, self._feature_ranges) for _ in range(n)])

    def _fitness(self, ruleset, X, y):
        preds = ruleset.predict(X)
        acc = accuracy_score(y, preds)
        complexity = sum(len(r.conditions) for r in ruleset.rules)
        return acc - self.complexity_penalty * complexity

    def _mutate(self, ind):
        rules = ind.rules[:]
        if random.random() < 0.5 and len(rules) > 1:
            del rules[random.randint(0, len(rules) - 1)]
        else:
            rules.append(Rule(self._n_features, self._feature_ranges))
        return RuleSet(rules)

    def _crossover(self, a, b):
        c1 = random.randint(0, len(a.rules))
        c2 = random.randint(0, len(b.rules))
        rules = a.rules[:c1] + b.rules[c2:]
        if not rules:
            rules = [Rule(self._n_features, self._feature_ranges)]
        return RuleSet(rules)

    def fit(self, X, y, feature_names=None, progress_callback=None):
        self._n_features = X.shape[1]
        self._feature_ranges = [
            (float(X[:, i].min()), float(X[:, i].max()))
            for i in range(self._n_features)
        ]
        self.feature_names = feature_names or [f'Merkmal_{i}' for i in range(self._n_features)]

        population = [self._new_ruleset() for _ in range(self.population_size)]
        self.history = []
        best_ever, best_score_ever = None, -np.inf

        for gen in range(self.generations):
            scored = sorted(
                [(ind, self._fitness(ind, X, y)) for ind in population],
                key=lambda x: x[1], reverse=True,
            )
            best, best_score = scored[0]
            self.history.append(round(best_score, 4))

            if best_score > best_score_ever:
                best_score_ever = best_score
                best_ever = best

            if progress_callback:
                progress_callback(gen + 1, self.generations, best_score)

            survivors = [ind for ind, _ in scored[:self.n_survivors]]
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                if random.random() < 0.5:
                    new_pop.append(self._mutate(random.choice(survivors)))
                else:
                    p1, p2 = random.sample(survivors, 2)
                    new_pop.append(self._crossover(p1, p2))
            population = new_pop

        self.best_model = best_ever
        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def get_rules(self):
        return [r.to_dict(self.feature_names) for r in self.best_model.rules]
