<?php
declare(strict_types=1);

class Rule
{
    public array $conditions;
    public int   $label;

    public function __construct(int $nFeatures, array $featureRanges)
    {
        $this->label      = rand(0, 1);
        $this->conditions = [];
        $nConds = rand(1, 3);
        for ($i = 0; $i < $nConds; $i++) {
            $f        = rand(0, $nFeatures - 1);
            [$lo, $hi] = $featureRanges[$f];
            $t        = $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
            $op       = rand(0, 1) ? '<' : '>';
            $this->conditions[] = [$f, $op, $t];
        }
    }

    public function matches(array $x): bool
    {
        foreach ($this->conditions as [$f, $op, $t]) {
            if ($op === '<' && $x[$f] >= $t) return false;
            if ($op === '>' && $x[$f] <= $t) return false;
        }
        return true;
    }

    public function toArray(array $featureNames): array
    {
        return [
            'bedingungen' => array_map(fn($c) => [
                'merkmal'     => $featureNames[$c[0]],
                'operator'    => $c[1],
                'schwellwert' => round($c[2], 4),
            ], $this->conditions),
            'klasse'      => $this->label,
            'klasse_text' => $this->label === 1 ? 'Steigt ↑' : 'Fällt ↓',
        ];
    }
}

class RuleSet
{
    /** @var Rule[] */
    public array $rules = [];

    public function predictOne(array $x): int
    {
        foreach ($this->rules as $rule) {
            if ($rule->matches($x)) return $rule->label;
        }
        return 0; // Standardklasse
    }

    public function predict(array $X): array
    {
        return array_map(fn($x) => $this->predictOne($x), $X);
    }
}

class GeneticRuleClassifier
{
    private int   $populationSize;
    private int   $generations;
    private int   $nSurvivors;
    private float $complexityPenalty;
    private int   $nFeatures    = 0;
    private array $featureRanges = [];

    public ?RuleSet $bestModel    = null;
    public array    $featureNames = [];
    public array    $history      = [];

    public function __construct(
        int   $populationSize    = 50,
        int   $generations       = 30,
        int   $nSurvivors        = 10,
        float $complexityPenalty = 0.01
    ) {
        $this->populationSize    = $populationSize;
        $this->generations       = $generations;
        $this->nSurvivors        = $nSurvivors;
        $this->complexityPenalty = $complexityPenalty;
    }

    private function newRuleSet(): RuleSet
    {
        $rs = new RuleSet();
        $n  = rand(1, 5);
        for ($i = 0; $i < $n; $i++) {
            $rs->rules[] = new Rule($this->nFeatures, $this->featureRanges);
        }
        return $rs;
    }

    private function fitness(RuleSet $rs, array $X, array $y): float
    {
        $preds   = $rs->predict($X);
        $n       = count($y);
        $correct = 0;
        for ($i = 0; $i < $n; $i++) {
            if ($preds[$i] === $y[$i]) $correct++;
        }
        $acc        = $n > 0 ? $correct / $n : 0.0;
        $complexity = array_sum(array_map(fn($r) => count($r->conditions), $rs->rules));
        return $acc - $this->complexityPenalty * $complexity;
    }

    private function mutate(RuleSet $ind): RuleSet
    {
        $rs        = new RuleSet();
        $rs->rules = $ind->rules;
        if ((mt_rand() / mt_getrandmax()) < 0.5 && count($rs->rules) > 1) {
            array_splice($rs->rules, rand(0, count($rs->rules) - 1), 1);
        } else {
            $rs->rules[] = new Rule($this->nFeatures, $this->featureRanges);
        }
        return $rs;
    }

    private function crossover(RuleSet $a, RuleSet $b): RuleSet
    {
        $rs        = new RuleSet();
        $c1        = rand(0, count($a->rules));
        $c2        = rand(0, count($b->rules));
        $rs->rules = array_merge(
            array_slice($a->rules, 0, $c1),
            array_slice($b->rules, $c2)
        );
        if (empty($rs->rules)) {
            $rs->rules[] = new Rule($this->nFeatures, $this->featureRanges);
        }
        return $rs;
    }

    public function fit(array $X, array $y, array $featureNames = [], ?callable $cb = null): void
    {
        $this->nFeatures   = count($X[0]);
        $this->featureNames = $featureNames ?: array_map(fn($i) => "Merkmal_$i", range(0, $this->nFeatures - 1));

        $this->featureRanges = [];
        for ($i = 0; $i < $this->nFeatures; $i++) {
            $col = array_column($X, $i);
            $this->featureRanges[] = [min($col), max($col)];
        }

        $population    = [];
        for ($i = 0; $i < $this->populationSize; $i++) {
            $population[] = $this->newRuleSet();
        }

        $this->history = [];
        $bestEver      = null;
        $bestEverScore = -PHP_INT_MAX;

        for ($gen = 0; $gen < $this->generations; $gen++) {
            $scored = array_map(fn($ind) => [$ind, $this->fitness($ind, $X, $y)], $population);
            usort($scored, fn($a, $b) => $b[1] <=> $a[1]);

            /** @var RuleSet $best */
            [$best, $bestScore] = $scored[0];
            $this->history[] = round((float) $bestScore, 4);

            if ($bestScore > $bestEverScore) {
                $bestEverScore = $bestScore;
                $bestEver      = $best;
            }

            if ($cb !== null) {
                $cb($gen + 1, $this->generations, (float) $bestScore);
            }

            $survivors = array_map(fn($s) => $s[0], array_slice($scored, 0, $this->nSurvivors));
            $newPop    = $survivors;

            while (count($newPop) < $this->populationSize) {
                if ((mt_rand() / mt_getrandmax()) < 0.5) {
                    $newPop[] = $this->mutate($survivors[array_rand($survivors)]);
                } else {
                    $keys = array_rand($survivors, 2);
                    $newPop[] = $this->crossover($survivors[$keys[0]], $survivors[$keys[1]]);
                }
            }

            $population = $newPop;
        }

        $this->bestModel = $bestEver;
    }

    public function predict(array $X): array
    {
        return $this->bestModel->predict($X);
    }

    public function getRules(): array
    {
        return array_map(fn($r) => $r->toArray($this->featureNames), $this->bestModel->rules);
    }
}
