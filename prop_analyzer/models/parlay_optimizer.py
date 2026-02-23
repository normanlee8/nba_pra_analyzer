# prop_analyzer/models/parlay_optimizer.py

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

# Underdog Fantasy Standard/Max Payout Multipliers
UNDERDOG_PAYOUTS = {
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 20.0,
    6: 25.0,  
    7: 40.0,  
    8: 80.0   
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
        
        # Memoization Cache: Stores Monte Carlo joint probabilities so we never simulate the same cluster twice
        self._simulation_cache = {}

    def _build_correlation_matrix(self) -> pd.DataFrame:
        """
        Builds a correlation matrix from historical box scores. 
        Groups by game_id to find same-game correlations between all stat categories, including PRA combos.
        """
        logger.info("Building correlation matrix from historical data...")
        
        required_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'Position', 'PTS', 'REB', 'AST']
        missing = [col for col in required_cols if col not in self.historical_data.columns]
        if missing:
            logger.warning(f"Missing columns for perfect correlation: {missing}. Returning empty matrix.")
            return pd.DataFrame()

        df = self.historical_data.copy()
        
        # Ensure base stats are numeric to safely calculate combos
        for stat in ['PTS', 'REB', 'AST']:
            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
        
        # NEW: Dynamically calculate combos so the matrix supports them
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        df['PR'] = df['PTS'] + df['REB']
        df['PA'] = df['PTS'] + df['AST']
        df['RA'] = df['REB'] + df['AST']
        
        # Create a unique key for Team + Position (e.g., LAL_PG)
        df['pos_stat'] = df['TEAM_ABBREVIATION'] + '_' + df['Position'] 
        
        categories = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']
        pivots = []
        
        for cat in categories:
            pivot = df.pivot_table(index='GAME_ID', columns='pos_stat', values=cat, aggfunc='mean').fillna(0)
            pivot.columns = [f"{c}_{cat}" for c in pivot.columns]
            pivots.append(pivot)
        
        # Merge and compute Pearson correlation
        combined = pd.concat(pivots, axis=1)
        corr_matrix = combined.corr()
        
        logger.info(f"Correlation matrix built successfully. Matrix shape: {corr_matrix.shape}")
        return corr_matrix

    def get_correlation(self, prop1: dict, prop2: dict) -> float:
        """
        Retrieves historical correlation between two props. Returns 0 if cross-game.
        """
        if prop1.get('game_id') != prop2.get('game_id'):
            return 0.0
            
        key1 = f"{prop1.get('team')}_{prop1.get('position')}_{prop1.get('stat_type', '').upper()}"
        key2 = f"{prop2.get('team')}_{prop2.get('position')}_{prop2.get('stat_type', '').upper()}"
        
        try:
            return self.correlation_matrix.loc[key1, key2]
        except KeyError:
            # Occurs if a position is missing from historical data
            return 0.0

    def simulate_same_game_cluster(self, cluster_props: list) -> float:
        """
        Uses a Gaussian Copula (Monte Carlo) to determine the joint probability.
        Utilizes caching to skip heavy computations for known clusters.
        """
        n = len(cluster_props)
        if n == 1:
            return cluster_props[0].get('win_prob', cluster_props[0].get('Prob', 0))
            
        # Create a unique, order-independent cache key for this exact cluster of props
        cache_key = frozenset([
            f"{p['player_name']}_{p.get('stat_type', p.get('Prop Category', ''))}_{p['pick']}" for p in cluster_props
        ])
        
        if cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]

        # Build covariance matrix
        cov_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.get_correlation(cluster_props[i], cluster_props[j])
                # Flip correlation if one is Over and the other is Under
                if cluster_props[i]['pick'] != cluster_props[j]['pick']:
                    corr = -corr
                cov_matrix[i, j] = corr
                cov_matrix[j, i] = corr
                
        # Mathematical safeguard to ensure the covariance matrix is positive semi-definite
        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        if min_eig < 0:
            cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)
                
        # Map individual win probabilities to standard normal thresholds
        thresholds = [norm.ppf(1 - prop.get('win_prob', prop.get('Prob', 0))) for prop in cluster_props]
        
        # Simulate Multivariant Normal Distribution
        mean = np.zeros(n)
        try:
            samples = np.random.multivariate_normal(mean, cov_matrix, self.num_simulations)
        except np.linalg.LinAlgError:
            # Fallback to independent probability if matrix still fails
            samples = np.random.multivariate_normal(mean, np.eye(n), self.num_simulations)
        
        # Calculate joint probability
        hits = np.all(samples > thresholds, axis=1)
        joint_prob = np.sum(hits) / self.num_simulations
        
        # Cache and return
        self._simulation_cache[cache_key] = joint_prob
        return joint_prob

    def calculate_ticket_ev(self, ticket: list) -> dict:
        """
        Calculates the true Expected Value (EV) of an Underdog parlay ticket.
        """
        num_legs = len(ticket)
        if num_legs not in UNDERDOG_PAYOUTS:
            return {'ev': -1, 'joint_prob': 0}
            
        # Group props by game to isolate correlations
        games = {}
        for prop in ticket:
            game_id = prop.get('game_id', 'unknown')
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(prop)
            
        # Compute total joint probability (cross-game = independent multiplication)
        total_joint_prob = 1.0
        for game_id, cluster in games.items():
            cluster_prob = self.simulate_same_game_cluster(cluster)
            total_joint_prob *= cluster_prob
            
        payout_multiplier = UNDERDOG_PAYOUTS[num_legs]
        expected_value = (total_joint_prob * payout_multiplier) - 1.0
        
        return {
            'ticket': ticket,
            'legs': num_legs,
            'joint_prob': total_joint_prob,
            'payout_multiplier': payout_multiplier,
            'ev': expected_value
        }

    def optimize_parlays(self, daily_props: list, min_legs=2, max_legs=8, top_n=10, beam_width=150) -> list:
        """
        Optimizes parlays using a Greedy Beam Search algorithm.
        Saves the best tickets PER LEG COUNT to ensure a diverse output.
        """
        logger.info(f"Optimizing parlays for {len(daily_props)} props using Beam Search...")
        
        # 1. Pre-Pruning: Keep props with high win probabilities, filtering out explicit traps.
        viable_props = []
        for p in daily_props:
            prob = p.get('win_prob', p.get('Prob', 0))
            tier = p.get('Tier', 'Pass')
            
            # Use raw win probability as the primary filter, but explicitly avoid known high variance traps
            if tier != 'Trap / High Variance' and prob >= 0.55:
                viable_props.append(p)
                
        # Take only the absolute best to keep combinatorics manageable and quality high
        viable_props = sorted(viable_props, key=lambda x: x.get('win_prob', x.get('Prob', 0)), reverse=True)[:35]
        
        logger.info(f"Filtered down to {len(viable_props)} core props for parlay construction based on Win Prob.")
        if len(viable_props) < min_legs:
            logger.warning("Not enough viable props to form profitable parlays today.")
            return []

        final_best_tickets = []
        current_beams = [[p] for p in viable_props]

        for k in range(2, max_legs + 1):
            logger.info(f"Evaluating {k}-leg combinations...")
            next_beams = []
            
            # Expand current top tickets by 1 leg
            for base_ticket in current_beams:
                existing_players = {p['player_name'] for p in base_ticket}
                
                for prop in viable_props:
                    if prop['player_name'] in existing_players:
                        continue 
                    
                    new_ticket = sorted(base_ticket + [prop], key=lambda x: x['player_name'])
                    ticket_signature = tuple(f"{p['player_name']}_{p.get('stat_type', p.get('Prop Category', ''))}_{p['pick']}" for p in new_ticket)
                    next_beams.append((ticket_signature, new_ticket))
            
            # Deduplicate the new combinations
            unique_next_beams = {}
            for sig, ticket in next_beams:
                if sig not in unique_next_beams:
                    unique_next_beams[sig] = ticket
                    
            # Evaluate all unique valid tickets at this leg count
            evaluated_tickets = []
            for ticket in unique_next_beams.values():
                ticket_eval = self.calculate_ticket_ev(ticket)
                if ticket_eval['ev'] > 0:
                    evaluated_tickets.append(ticket_eval)
            
            if not evaluated_tickets:
                logger.info(f"No +EV paths remaining at {k} legs. Halting search.")
                break

            # SAVE THE BEST FOR THIS SPECIFIC LEG COUNT
            leg_best = sorted(evaluated_tickets, key=lambda x: x['joint_prob'], reverse=True)[:top_n]
            final_best_tickets.extend(leg_best)
            
            # THE GREEDY CUT FOR THE NEXT BEAM
            evaluated_tickets = sorted(evaluated_tickets, key=lambda x: x['ev'], reverse=True)[:beam_width]
            current_beams = [t['ticket'] for t in evaluated_tickets]

        return final_best_tickets