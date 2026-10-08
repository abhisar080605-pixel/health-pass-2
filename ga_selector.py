"""
HealthPass - Adaptive Genetic Algorithm (AGA-Lite) Feature Selector

Inspired by Abdollahi et al. (2026), "Adaptive feature optimization in deep
neural architectures for intelligent diabetes prediction" (Intelligence-Based
Medicine 15, 100393).

Core Concepts:
- Binary chromosome representation: bit i = 1 includes feature i, 0 excludes it.
- Fitness function:
    fitness(mask) = w1 * val_accuracy + w2 * (1 - (selected_features / total_features))
    Default weights: w1 = 0.8, w2 = 0.2.
- Genetic Operators:
    - Tournament selection (size k=3)
    - One-point crossover (rate ~0.8)
    - Adaptive bit-flip mutation (scales when diversity drops)
    - Elitism: top 2 chromosomes survive unconditionally each generation
    - Early stopping when best fitness plateaus for N generations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import io
import base64

class AdaptiveGeneticSelector(BaseEstimator, TransformerMixin):
    """
    Adaptive Genetic Algorithm for feature selection in clinical risk screening.
    """
    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_population: int = 30,
        n_generations: int = 35,
        w1: float = 0.8,
        w2: float = 0.2,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: Optional[float] = None,
        elitism_count: int = 2,
        early_stopping_patience: int = 8,
        cv: int = 3,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.estimator = estimator
        self.n_population = n_population
        self.n_generations = n_generations
        self.w1 = w1
        self.w2 = w2
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.early_stopping_patience = early_stopping_patience
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.best_chromosome_: Optional[np.ndarray] = None
        self.best_fitness_: float = 0.0
        self.best_val_accuracy_: float = 0.0
        self.best_features_: List[str] = []
        self.pruned_features_: List[str] = []
        self.all_features_: List[str] = []
        self.history_: List[Dict] = []
        self._fitness_cache: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    def _init_population(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """Initialize population with uniform random binary chromosomes."""
        population = rng.integers(0, 2, size=(self.n_population, n_features))
        # Ensure every chromosome has at least 1 feature selected
        for i in range(self.n_population):
            if np.sum(population[i]) == 0:
                population[i, rng.integers(0, n_features)] = 1
        return population

    def _evaluate_chromosome(
        self,
        chromosome: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        clf: BaseEstimator
    ) -> Tuple[float, float]:
        """
        Evaluate fitness = w1 * val_accuracy + w2 * (1 - selected/total).
        Uses caching to avoid re-evaluating identical chromosomes.
        """
        key = tuple(chromosome.tolist())
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        selected_idx = np.where(chromosome == 1)[0]
        n_features = len(chromosome)
        n_selected = len(selected_idx)

        # Severe penalty for selecting zero features
        if n_selected == 0:
            return 0.0, 0.0

        X_subset = X[:, selected_idx]
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(clf, X_subset, y, cv=skf, scoring="accuracy", n_jobs=1)
        val_acc = float(np.mean(scores))

        parsimony = 1.0 - (n_selected / n_features)
        fitness = (self.w1 * val_acc) + (self.w2 * parsimony)

        result = (float(fitness), val_acc)
        self._fitness_cache[key] = result
        return result

    def _tournament_select(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Select best candidate from random tournament sample."""
        sample_indices = rng.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = sample_indices[np.argmax(fitnesses[sample_indices])]
        return population[best_idx].copy()

    def _one_point_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one-point crossover between parents."""
        if rng.random() > self.crossover_rate or len(parent1) < 2:
            return parent1.copy(), parent2.copy()
        
        point = rng.integers(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _bit_flip_mutation(
        self,
        chromosome: np.ndarray,
        mutation_rate: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Mutate bits with probability mutation_rate."""
        mutated = chromosome.copy()
        flip_mask = rng.random(len(mutated)) < mutation_rate
        mutated[flip_mask] = 1 - mutated[flip_mask]
        
        # Ensure at least one feature remains selected
        if np.sum(mutated) == 0:
            mutated[rng.integers(0, len(mutated))] = 1
        return mutated

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """
        Run the Adaptive Genetic Algorithm feature optimization.
        """
        if isinstance(X, pd.DataFrame):
            self.all_features_ = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            self.all_features_ = feature_names or [f"f_{i}" for i in range(X_mat.shape[1])]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y).ravel()
        else:
            y_arr = np.asarray(y).ravel()

        n_samples, n_features = X_mat.shape
        rng = np.random.default_rng(self.random_state)
        base_mutation_rate = self.mutation_rate or (1.0 / n_features)

        # Use fast, lightweight classifier for GA search
        eval_clf = self.estimator or RandomForestClassifier(
            n_estimators=45,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=1
        )

        population = self._init_population(n_features, rng)
        self.history_ = []
        self._fitness_cache.clear()

        best_overall_fitness = -1.0
        best_overall_chromosome = None
        best_overall_acc = 0.0
        generations_without_improvement = 0

        for gen in range(1, self.n_generations + 1):
            # 1. Evaluate fitness of all individuals
            fitness_results = [
                self._evaluate_chromosome(ind, X_mat, y_arr, eval_clf)
                for ind in population
            ]
            fitnesses = np.array([res[0] for res in fitness_results])
            accuracies = np.array([res[1] for res in fitness_results])

            # Current generation stats
            best_gen_idx = int(np.argmax(fitnesses))
            best_gen_fitness = float(fitnesses[best_gen_idx])
            avg_gen_fitness = float(np.mean(fitnesses))
            best_gen_acc = float(accuracies[best_gen_idx])
            best_gen_selected = int(np.sum(population[best_gen_idx]))

            # Check overall best
            if best_gen_fitness > best_overall_fitness:
                best_overall_fitness = best_gen_fitness
                best_overall_chromosome = population[best_gen_idx].copy()
                best_overall_acc = best_gen_acc
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Population diversity check (fraction of unique chromosomes)
            unique_count = len({tuple(ind.tolist()) for ind in population})
            diversity = unique_count / self.n_population

            # Adaptive mutation rate: boost mutation if diversity is dropping or stagnant
            if diversity < 0.4 or generations_without_improvement >= 3:
                curr_mutation_rate = min(0.35, base_mutation_rate * 2.2)
            else:
                curr_mutation_rate = base_mutation_rate

            self.history_.append({
                "generation": gen,
                "best_fitness": round(best_gen_fitness, 4),
                "avg_fitness": round(avg_gen_fitness, 4),
                "best_accuracy": round(best_gen_acc, 4),
                "selected_count": best_gen_selected,
                "total_features": n_features,
                "diversity": round(diversity, 3),
                "mutation_rate": round(curr_mutation_rate, 4),
                "best_chromosome": population[best_gen_idx].tolist()
            })

            if self.verbose:
                print(
                    f"Gen {gen:02d}/{self.n_generations:02d} | "
                    f"Best Fit: {best_gen_fitness:.4f} (Acc: {best_gen_acc:.3f}, Feats: {best_gen_selected}/{n_features}) | "
                    f"Avg Fit: {avg_gen_fitness:.4f} | Div: {diversity:.2f}"
                )

            # Early stopping check
            if generations_without_improvement >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping triggered at generation {gen} (no improvement for {self.early_stopping_patience} gens).")
                break

            # 2. Elitism: preserve top individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            next_population = [population[idx].copy() for idx in sorted_indices[:self.elitism_count]]

            # 3. Create rest of next generation via Selection, Crossover, and Mutation
            while len(next_population) < self.n_population:
                parent1 = self._tournament_select(population, fitnesses, rng)
                parent2 = self._tournament_select(population, fitnesses, rng)
                child1, child2 = self._one_point_crossover(parent1, parent2, rng)
                child1 = self._bit_flip_mutation(child1, curr_mutation_rate, rng)
                next_population.append(child1)
                if len(next_population) < self.n_population:
                    child2 = self._bit_flip_mutation(child2, curr_mutation_rate, rng)
                    next_population.append(child2)

            population = np.array(next_population)

        self.best_chromosome_ = best_overall_chromosome
        self.best_fitness_ = best_overall_fitness
        self.best_val_accuracy_ = best_overall_acc

        selected_indices = np.where(self.best_chromosome_ == 1)[0]
        self.best_features_ = [self.all_features_[i] for i in selected_indices]
        self.pruned_features_ = [self.all_features_[i] for i in range(n_features) if i not in selected_indices]

        if self.verbose:
            print("\n" + "="*50)
            print(f"GA Feature Selection Complete!")
            print(f"Optimal features ({len(self.best_features_)}/{n_features}): {self.best_features_}")
            print(f"Pruned uninformative features ({len(self.pruned_features_)}): {self.pruned_features_}")
            print(f"Best Fitness: {self.best_fitness_:.4f} (CV Acc: {self.best_val_accuracy_:.3f})")
            print("="*50 + "\n")

        return self

    def transform(self, X):
        """Transform X to retain only the GA-selected features."""
        if self.best_chromosome_ is None:
            raise ValueError("AdaptiveGeneticSelector must be fitted before transforming data.")
        
        selected_indices = np.where(self.best_chromosome_ == 1)[0]
        if isinstance(X, pd.DataFrame):
            return X[self.best_features_]
        else:
            return np.asarray(X)[:, selected_indices]

    def get_support(self) -> np.ndarray:
        """Return boolean mask of selected features."""
        if self.best_chromosome_ is None:
            raise ValueError("Selector is not fitted yet.")
        return self.best_chromosome_ == 1

    def plot_convergence(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot GA fitness convergence and feature count reduction.
        Optionally saves to save_path or returns base64 PNG string.
        """
        if not self.history_:
            return None

        gens = [h["generation"] for h in self.history_]
        best_fits = [h["best_fitness"] for h in self.history_]
        avg_fits = [h["avg_fitness"] for h in self.history_]
        selected_counts = [h["selected_count"] for h in self.history_]

        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)

        color1 = "#1f77b4"
        color2 = "#aec7e8"
        ax1.set_xlabel("Generation", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Fitness Score", color=color1, fontsize=11, fontweight="bold")
        l1 = ax1.plot(gens, best_fits, color=color1, marker="o", linewidth=2.2, label="Best Fitness")
        l2 = ax1.plot(gens, avg_fits, color=color2, linestyle="--", linewidth=1.8, label="Average Fitness")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2 = ax1.twinx()
        color3 = "#d62728"
        ax2.set_ylabel("Features Selected", color=color3, fontsize=11, fontweight="bold")
        l3 = ax2.plot(gens, selected_counts, color=color3, marker="s", linestyle="-.", linewidth=1.8, label="Feature Count")
        ax2.tick_params(axis="y", labelcolor=color3)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower right", framealpha=0.9)
        plt.title("HealthPass Adaptive GA Feature Optimization Convergence\n(w1=0.8 Acc, w2=0.2 Parsimony)", fontsize=12, pad=12)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            return save_path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
