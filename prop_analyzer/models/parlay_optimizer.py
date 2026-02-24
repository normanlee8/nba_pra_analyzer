# prop_analyzer/models/parlay_optimizer.py

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

# Underdog Fantasy Standard/Max Payout Multipliers (Kept for reference, but no longer used for optimization)
UNDERDOG_PAYOUTS = {
    2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 25.0, 7: 40.0, 8: 80.0   
}

class ParlayOptimizer:
    def __init__(self, historical_data: pd.DataFrame, num_simulations: int = 10000):
        """
        Initializes the Parlay Optimizer.
        Builds the historical correlation matrix upon initialization.
        """
        self.num_simulations = num_simulations
        self.historical_data = historical_data
        self.correlation_matrix = self._build_correlation_matrix()
        
        # Memoization Cache
        self._simulation_cache = {}

    def _build_correlation_matrix(self) -> pd.DataFrame:
        """
        Builds a correlation matrix from historical box scores. 
        Groups by game_id to find same-game correlations between all stat categories.
        """
        logger.info("Building correlation matrix from historical data...")
        
        required_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'Position', 'PTS', 'REB', 'AST']
        missing = [col for col in required_cols if col not in self.historical_data.columns]
        if missing:
            logger.warning(f"Missing columns for perfect correlation: {missing}. Returning empty matrix.")
            return pd.DataFrame()

        df = self.historical_data.copy()
        
        for stat in ['PTS', 'REB', 'AST']:
            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
        
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        df['PR'] = df['PTS'] + df['REB']
        df['PA'] = df['PTS'] + df['AST']
        df['RA'] = df['REB'] + df['AST']
        
        df['pos_stat'] = df['TEAM_ABBREVIATION'] + '_' + df['Position'] 
        
        categories = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']
        pivots = []
        
        for cat in categories:
            pivot = df.pivot_table(index='GAME_ID', columns='pos_stat', values=cat, aggfunc='mean').fillna(0)
            pivot.columns = [f"{c}_{cat}" for c in pivot.columns]
            pivots.append(pivot)
        
        combined = pd.concat(pivots, axis=1)
        corr_matrix = combined.corr()
        
        logger.info(f"Correlation matrix built successfully. Matrix shape: {corr_matrix.shape}")
        return corr_matrix

    def get_correlation(self, prop1: dict, prop2: dict) -> float:
        if prop1.get('game_id') != prop2.get('game_id'):
            return 0.0
            
        key1 = f"{prop1.get('team')}_{prop1.get('position')}_{prop1.get('stat_type', '').upper()}"
        key2 = f"{prop2.get('team')}_{prop2.get('position')}_{prop2.get('stat_type', '').upper()}"
        
        try:
            return self.correlation_matrix.loc[key1, key2]
        except KeyError:
            return 0.0

    def simulate_same_game_cluster(self, cluster_props: list) -> float:
        """
        Uses a Gaussian Copula (Monte Carlo) to determine the true joint probability 
        of correlated same-game events.
        """
        n = len(cluster_props)
        if n == 1:
            return cluster_props[0].get('win_prob', cluster_props[0].get('Prob', 0))
            
        cache_key = frozenset([
            f"{p['player_name']}_{p.get('stat_type', p.get('Prop Category', ''))}_{p['pick']}" for p in cluster_props
        ])
        
        if cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]

        cov_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.get_correlation(cluster_props[i], cluster_props[j])
                if cluster_props[i]['pick'] != cluster_props[j]['pick']:
                    corr = -corr
                cov_matrix[i, j] = corr
                cov_matrix[j, i] = corr
                
        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        if min_eig < 0:
            cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)
                
        thresholds = [norm.ppf(1 - prop.get('win_prob', prop.get('Prob', 0))) for prop in cluster_props]
        mean = np.zeros(n)
        try:
            samples = np.random.multivariate_normal(mean, cov_matrix, self.num_simulations)
        except np.linalg.LinAlgError:
            samples = np.random.multivariate_normal(mean, np.eye(n), self.num_simulations)
        
        hits = np.all(samples > thresholds, axis=1)
        joint_prob = np.sum(hits) / self.num_simulations
        
        self._simulation_cache[cache_key] = joint_prob
        return joint_prob

    def calculate_ticket_metrics(self, ticket: list) -> dict:
        """
        Calculates the true Joint Probability of the parlay hitting.
        EV has been completely removed.
        """
        num_legs = len(ticket)
            
        games = {}
        for prop in ticket:
            game_id = prop.get('game_id', 'unknown')
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(prop)
            
        total_joint_prob = 1.0
        for game_id, cluster in games.items():
            cluster_prob = self.simulate_same_game_cluster(cluster)
            total_joint_prob *= cluster_prob
            
        payout_multiplier = UNDERDOG_PAYOUTS.get(num_legs, 0.0)
        
        return {
            'ticket': ticket,
            'legs': num_legs,
            'joint_prob': total_joint_prob,
            'payout_multiplier': payout_multiplier
        }

    def optimize_parlays(self, daily_props: list, min_legs=2, max_legs=8, top_n=10, beam_width=150) -> list:
        """
        Optimizes parlays using a Greedy Beam Search algorithm.
        Now strictly maximizes Joint Probability.
        """
        logger.info(f"Optimizing parlays for {len(daily_props)} props using Probability Maximization...")
        
        # 1. Strict Probability Pruning
        viable_props = []
        for p in daily_props:
            prob = p.get('win_prob', p.get('Prob', 0))
            tier = p.get('Tier', 'Pass')
            
            # Require minimum 58% baseline probability to even be considered for a parlay
            if tier not in ['Trap / High Variance', 'Pass / Too Volatile', 'Pass'] and prob >= 0.58:
                viable_props.append(p)
                
        viable_props = sorted(viable_props, key=lambda x: x.get('win_prob', x.get('Prob', 0)), reverse=True)[:35]
        
        logger.info(f"Filtered down to {len(viable_props)} highly consistent props for parlay construction.")
        if len(viable_props) < min_legs:
            logger.warning("Not enough viable props to form high-probability parlays today.")
            return []

        final_best_tickets = []
        current_beams = [[p] for p in viable_props]

        for k in range(2, max_legs + 1):
            logger.info(f"Evaluating {k}-leg combinations...")
            next_beams = []
            
            for base_ticket in current_beams:
                existing_players = {p['player_name'] for p in base_ticket}
                
                for prop in viable_props:
                    if prop['player_name'] in existing_players:
                        continue 
                    
                    new_ticket = sorted(base_ticket + [prop], key=lambda x: x['player_name'])
                    ticket_signature = tuple(f"{p['player_name']}_{p.get('stat_type', p.get('Prop Category', ''))}_{p['pick']}" for p in new_ticket)
                    next_beams.append((ticket_signature, new_ticket))
            
            unique_next_beams = {}
            for sig, ticket in next_beams:
                if sig not in unique_next_beams:
                    unique_next_beams[sig] = ticket
                    
            evaluated_tickets = []
            for ticket in unique_next_beams.values():
                ticket_eval = self.calculate_ticket_metrics(ticket)
                # Keep ticket if the joint probability is non-zero
                if ticket_eval['joint_prob'] > 0:
                    evaluated_tickets.append(ticket_eval)
            
            if not evaluated_tickets:
                break

            # Sort strictly by Highest Likelihood of Hitting
            leg_best = sorted(evaluated_tickets, key=lambda x: x['joint_prob'], reverse=True)[:top_n]
            final_best_tickets.extend(leg_best)
            
            # Feed the highest probability tickets to the next beam level
            evaluated_tickets = sorted(evaluated_tickets, key=lambda x: x['joint_prob'], reverse=True)[:beam_width]
            current_beams = [t['ticket'] for t in evaluated_tickets]

        return final_best_tickets