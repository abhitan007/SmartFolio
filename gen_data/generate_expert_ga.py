"""
Enhanced Expert Generation with Genetic Algorithm for Stock Selection
Adapted for SmartFolio's multi-hot binary action space
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class RiskCategory:
    """Risk categories with fitness function weights"""
    
    CONSERVATIVE = {
        'name': 'conservative',
        'weights': {
            'return': 0.10,
            'volatility': -0.35,  # Heavy penalty
            'diversification': 0.25,
            'correlation': -0.15,
            'quality': 0.15
        },
        'target_stocks_range': (10, 15)
    }
    
    MODERATE = {
        'name': 'moderate',
        'weights': {
            'return': 0.25,
            'volatility': -0.20,
            'diversification': 0.25,
            'correlation': -0.15,
            'quality': 0.10
        },
        'target_stocks_range': (10, 15)
    }
    
    AGGRESSIVE = {
        'name': 'aggressive',
        'weights': {
            'return': 0.40,  # High focus on returns
            'volatility': -0.10,  # Low penalty
            'diversification': 0.20,
            'correlation': -0.10,
            'quality': 0.15
        },
        'target_stocks_range': (10, 15)
    }


class StockSelectionGA:
    """Genetic Algorithm for multi-hot binary stock selection"""
    
    def __init__(self,
                 risk_category: Dict,
                 population_size: int = 50,
                 n_generations: int = 30,
                 mutation_rate: float = 0.05,
                 elite_ratio: float = 0.2):
        self.risk_category = risk_category
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = int(population_size * elite_ratio)
        
        self.fitness_weights = risk_category['weights']
        self.min_stocks, self.max_stocks = risk_category['target_stocks_range']
    
    def calculate_fitness_metrics(self,
                                  selected_indices: List[int],
                                  returns: np.ndarray,
                                  correlation_matrix: np.ndarray,
                                  industry_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate fitness metrics for selected stocks"""
        if len(selected_indices) == 0:
            return {k: 0.0 for k in self.fitness_weights.keys()}
        
        # 1. Return Potential
        return_metric = np.mean(returns[selected_indices])
        
        # 2. Volatility (estimated from recent returns if available)
        volatility_metric = np.std(returns[selected_indices])
        
        # 3. Diversification (sector balance using Herfindahl index)
        sector_counts = {}
        for idx in selected_indices:
            sector_stocks = np.where(industry_matrix[idx] > 0)[0]
            sector_id = tuple(sorted(sector_stocks))
            sector_counts[sector_id] = sector_counts.get(sector_id, 0) + 1
        
        if sector_counts:
            sector_weights = np.array(list(sector_counts.values())) / len(selected_indices)
            herfindahl = np.sum(sector_weights ** 2)
            diversification_metric = 1 - herfindahl
        else:
            diversification_metric = 0.5
        
        # 4. Correlation (lower is better)
        if len(selected_indices) > 1:
            selected_corr = correlation_matrix[np.ix_(selected_indices, selected_indices)]
            upper_tri_idx = np.triu_indices_from(selected_corr, k=1)
            correlation_metric = np.mean(np.abs(selected_corr[upper_tri_idx]))
        else:
            correlation_metric = 0.0
        
        # 5. Quality Score (Sharpe-like ratio)
        if volatility_metric > 1e-6:
            quality_metric = return_metric / volatility_metric
        else:
            quality_metric = 0.0
        
        return {
            'return': return_metric,
            'volatility': volatility_metric,
            'diversification': diversification_metric,
            'correlation': correlation_metric,
            'quality': quality_metric
        }
    
    def evaluate_chromosome(self,
                           chromosome: np.ndarray,
                           returns: np.ndarray,
                           correlation_matrix: np.ndarray,
                           industry_matrix: np.ndarray) -> float:
        """Evaluate fitness of a chromosome (binary stock selection)"""
        selected_indices = np.where(chromosome == 1)[0]
        n_selected = len(selected_indices)
        
        # Hard constraint: Must select correct number of stocks
        if n_selected < self.min_stocks or n_selected > self.max_stocks:
            return -np.inf
        
        # Calculate metrics
        metrics = self.calculate_fitness_metrics(
            selected_indices.tolist(),
            returns,
            correlation_matrix,
            industry_matrix
        )
        
        # Weighted fitness
        fitness = sum(
            self.fitness_weights[key] * metrics[key]
            for key in self.fitness_weights.keys()
        )
        
        return fitness
    
    def initialize_population(self, n_stocks: int) -> List[np.ndarray]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            n_select = np.random.randint(self.min_stocks, self.max_stocks + 1)
            chromosome = np.zeros(n_stocks, dtype=int)
            selected_idx = np.random.choice(n_stocks, size=n_select, replace=False)
            chromosome[selected_idx] = 1
            population.append(chromosome)
        
        return population
    
    def tournament_selection(self,
                            population: List[np.ndarray],
                            fitness_scores: np.ndarray,
                            tournament_size: int = 3) -> np.ndarray:
        """Tournament selection"""
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        best_idx = indices[np.argmax(fitness_scores[indices])]
        return population[best_idx].copy()
    
    def uniform_crossover(self,
                         parent1: np.ndarray,
                         parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover"""
        mask = np.random.rand(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit flip mutation with constraint enforcement"""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Enforce constraints
        n_selected = np.sum(mutated)
        
        if n_selected < self.min_stocks:
            n_to_add = self.min_stocks - n_selected
            available = np.where(mutated == 0)[0]
            if len(available) >= n_to_add:
                add_idx = np.random.choice(available, size=n_to_add, replace=False)
                mutated[add_idx] = 1
        
        elif n_selected > self.max_stocks:
            n_to_remove = n_selected - self.max_stocks
            selected = np.where(mutated == 1)[0]
            remove_idx = np.random.choice(selected, size=n_to_remove, replace=False)
            mutated[remove_idx] = 0
        
        return mutated
    
    def evolve(self,
              returns: np.ndarray,
              correlation_matrix: np.ndarray,
              industry_matrix: np.ndarray,
              verbose: bool = False) -> Tuple[np.ndarray, float]:
        """Run GA evolution"""
        n_stocks = len(returns)
        population = self.initialize_population(n_stocks)
        
        best_overall_fitness = -np.inf
        best_overall_chromosome = None
        
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = np.array([
                self.evaluate_chromosome(chrom, returns, correlation_matrix, industry_matrix)
                for chrom in population
            ])
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_chromosome = population[gen_best_idx].copy()
            
            # Evolution
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.uniform_crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population[:self.population_size]
        
        return best_overall_chromosome, best_overall_fitness


def generate_expert_strategy_ga(returns: np.ndarray,
                                industry_relation_matrix: np.ndarray,
                                correlation_matrix: np.ndarray,
                                risk_category: str = 'moderate',
                                ga_generations: int = 30,
                                verbose: bool = False) -> np.ndarray:
    """
    Generate expert portfolio using GA for stock selection
    Returns: Multi-hot binary vector (compatible with your action space)
    """
    # Select risk category
    if risk_category == 'conservative':
        risk_config = RiskCategory.CONSERVATIVE
    elif risk_category == 'aggressive':
        risk_config = RiskCategory.AGGRESSIVE
    else:
        risk_config = RiskCategory.MODERATE
    
    # Run GA
    ga = StockSelectionGA(
        risk_category=risk_config,
        population_size=50,
        n_generations=ga_generations,
        mutation_rate=0.05
    )
    
    best_chromosome, best_fitness = ga.evolve(
        returns=returns,
        correlation_matrix=correlation_matrix,
        industry_matrix=industry_relation_matrix,
        verbose=verbose
    )
    
    return best_chromosome  # Multi-hot binary vector


def generate_expert_trajectories_ga(args,
                                   dataset,
                                   num_trajectories: int = 100,
                                   risk_category: str = 'moderate',
                                   ga_generations: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate expert trajectories using GA
    Output format matches your IRL framework: (state, action_multi_hot)
    """
    expert_trajectories = []
    industry_relation_matrix = load_industry_relation_matrix(args.market)
    
    if industry_relation_matrix is None:
        print("Warning: Industry matrix not found")
        n_stocks = len(dataset[0]['labels'].numpy())
        industry_relation_matrix = np.eye(n_stocks)
    
    # Determine risk categories
    if risk_category == 'mixed':
        risk_categories = ['conservative', 'moderate', 'aggressive']
    else:
        risk_categories = [risk_category]
    
    print(f"\nGenerating {num_trajectories} GA expert trajectories")
    print(f"Risk categories: {', '.join(risk_categories)}")
    print("="*60)
    
    for traj_idx in range(num_trajectories):
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]
        
        features = data['features'].numpy()
        correlation_matrix = data['corr'].numpy()
        ind_matrix = data['industry_matrix'].numpy()
        pos_matrix = data['pos_matrix'].numpy()
        neg_matrix = data['neg_matrix'].numpy()
        returns = data['labels'].numpy()
        
        # Select risk category
        current_risk = np.random.choice(risk_categories) if risk_category == 'mixed' else risk_category
        
        try:
            # GA-based expert (returns multi-hot binary)
            expert_actions = generate_expert_strategy_ga(
                returns=returns,
                industry_relation_matrix=ind_matrix,
                correlation_matrix=correlation_matrix,
                risk_category=current_risk,
                ga_generations=ga_generations,
                verbose=False
            )
        except Exception as e:
            print(f"Warning: Trajectory {traj_idx} failed: {e}")
            # Fallback: top returns
            n_select = min(12, len(returns))
            expert_actions = np.zeros(len(returns), dtype=int)
            top_indices = np.argsort(-returns)[:n_select]
            expert_actions[top_indices] = 1
        
        # Create flattened state (matching new observation format)
        state_parts = []
        if args.ind_yn:
            state_parts.append(ind_matrix.flatten())
        else:
            state_parts.append(np.zeros(ind_matrix.size))
        if args.pos_yn:
            state_parts.append(pos_matrix.flatten())
        else:
            state_parts.append(np.zeros(pos_matrix.size))
        if args.neg_yn:
            state_parts.append(neg_matrix.flatten())
        else:
            state_parts.append(np.zeros(neg_matrix.size))
        
        state_parts.append(features.flatten())
        state = np.concatenate(state_parts)
        
        expert_trajectories.append((state, expert_actions))
        
        if (traj_idx + 1) % 100 == 0:
            print(f"  Generated {traj_idx + 1}/{num_trajectories} trajectories")
    
    print(f"\n✓ Generated {len(expert_trajectories)} trajectories")
    
    # Statistics
    avg_selected = np.mean([np.sum(a) for _, a in expert_trajectories])
    print(f"  Average stocks selected: {avg_selected:.1f}")
    
    return expert_trajectories


def load_industry_relation_matrix(market: str) -> Optional[np.ndarray]:
    """Load industry relation matrix"""
    try:
        with open(f"dataset_default/data_train_predict_{market}/industry.npy", 'rb') as f:
            return np.load(f)
    except FileNotFoundError:
        return None


def save_expert_trajectories(trajectories: List, save_path: str):
    """Save expert trajectories"""
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"✓ Saved to {save_path}")


if __name__ == '__main__':
    class Args:
        market = 'hs300'
        input_dim = 6
        ind_yn = True
        pos_yn = True
        neg_yn = True
    
    args = Args()
    
    try:
        from dataloader.data_loader import AllGraphDataSampler
        
        data_dir = f'dataset_default/data_train_predict_{args.market}/1_hy/'
        train_dataset = AllGraphDataSampler(
            base_dir=data_dir,
            date=True,
            train_start_date='2019-01-02',
            train_end_date='2022-12-30',
            mode="train"
        )
        
        print(f"\n{'='*70}")
        print("GA-ENHANCED EXPERT GENERATION FOR SMARTFOLIO")
        print(f"{'='*70}")
        
        expert_trajectories = generate_expert_trajectories_ga(
            args,
            train_dataset,
            num_trajectories=1000,
            risk_category='mixed',
            ga_generations=30
        )
        
        save_path = f'dataset_default/expert_trajectories_{args.market}_ga.pkl'
        save_expert_trajectories(expert_trajectories, save_path)
        
        print(f"\n{'='*70}")
        print("SUCCESS!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
