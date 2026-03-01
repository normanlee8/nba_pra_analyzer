# prop_analyzer/models/parlay_optimizer.py

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

# Underdog Fantasy Standard/Max Payout Multipliers
UNDERDOG_PAYOUTS = {
    2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 25.0, 7: 40.0, 8: 80.0   
}

class ParlayOptimizer:
    def __init__(self, historical_data: pd.DataFrame = None, num_simulations: int = 10000):
        """
        Initializes the Parlay Optimizer.
        Historical data parameter kept for API compatibility, but correlation 
        now relies on structural NBA covariance rules for mathematical stability.
        """
        self.num_simulations = num_simulations
        self._simulation_cache = {}
        logger.info("Initialized ParlayOptimizer with Structural Same-Game Correlation Rules.")

    def get_correlation(self, prop1: dict, prop2: dict) -> float:
        """
        Determines the structural correlation between two props in the same game.
        """
        if prop1.get('game_id') != prop2.get('game_id'):
            return 0.0
            
        is_same_team = prop1.get('team') == prop2.get('team')
        
        # Map combo stats to primary drivers for correlation purposes
        def get_base_stat(stat):
            stat = str(stat).upper()
            if stat in ['PRA', 'PR', 'PA', 'PTS']: return 'PTS'
            if stat in ['RA', 'REB']: return 'REB'
            return stat
            
        base1 = get_base_stat(prop1.get('stat_type'))
        base2 = get_base_stat(prop2.get('stat_type'))
        
        # DFS Same-Game Covariance Rules
        if is_same_team:
            if (base1 == 'PTS' and base2 == 'AST') or (base1 == 'AST' and base2 == 'PTS'):
                return 0.25  # Scoring directly correlates with teammate assists
            if base1 == 'REB' and base2 == 'REB':
                return -0.20 # Rebounds are zero-sum among teammates
            if base1 == 'PTS' and base2 == 'PTS':
                return -0.15 # Usage sharing (only one ball)
            if base1 == 'AST' and base2 == 'AST':
                return -0.10 # Primary vs Secondary handler sharing
        else:
            if base1 == 'PTS' and base2 == 'PTS':
                return 0.20  # Fast pace / shootout environment
            if base1 == 'REB' and base2 == 'REB':
                return -0.15 # Zero-sum total available rebounds
            if base1 == 'AST' and base2 == 'AST':
                return 0.15  # Pace up translates to more assists for both sides
            if (base1 == 'PTS' and base2 == 'REB') or (base1 == 'REB' and base2 == 'PTS'):
                return -0.10 # Opponent scoring heavily reduces defensive rebound opportunities

        return 0.0

    def simulate_same_game_cluster(self, cluster_props: list) -> float:
        """
        Uses a Gaussian Copula (Monte Carlo) to determine the true joint probability.
        Properly maintains Standard Normal diagonals.
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
                # Flip correlation mathematically if pick directions differ (e.g. Over vs Under)
                if cluster_props[i]['pick'] != cluster_props[j]['pick']:
                    corr = -corr
                cov_matrix[i, j] = corr
                cov_matrix[j, i] = corr
                
        # PSD Check - Ensure matrix is positive semi-definite without ruining diagonal variances
        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        while min_eig < 0:
            # Shrink off-diagonals toward 0 (independence) by blending with the Identity matrix
            cov_matrix = (cov_matrix * 0.8) + (np.eye(n) * 0.2)
            min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
                
        thresholds = [norm.ppf(1 - prop.get('win_prob', prop.get('Prob', 0))) for prop in cluster_props]
        mean = np.zeros(n)
        
        try:
            samples = np.random.multivariate_normal(mean, cov_matrix, self.num_simulations)
            hits = np.all(samples > thresholds, axis=1)
            joint_prob = np.sum(hits) / self.num_simulations
        except np.linalg.LinAlgError:
            # Fallback to independence if covariance matrix fundamentally fails
            joint_prob = np.prod([prop.get('win_prob', prop.get('Prob', 0)) for prop in cluster_props])
        
        self._simulation_cache[cache_key] = joint_prob
        return joint_prob

    def calculate_ticket_metrics(self, ticket: list) -> dict:
        """
        Calculates the true Joint Probability of the parlay hitting and enforces strict DFS rules.
        Now calculates Expected Value (EV) so larger parlays aren't punished in sorting.
        """
        num_legs = len(ticket)
            
        # DFS Rule Enforcement: Tickets MUST have players from at least 2 different teams
        unique_teams = {prop.get('team') for prop in ticket}
        if len(unique_teams) < 2:
            return {
                'ticket': ticket,
                'legs': num_legs,
                'joint_prob': 0.0,
                'payout_multiplier': 0.0,
                'expected_value': 0.0
            }

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
        
        # CRITICAL FIX: Calculate EV to compare parlays of different sizes fairly
        expected_value = total_joint_prob * payout_multiplier
        
        return {
            'ticket': ticket,
            'legs': num_legs,
            'joint_prob': total_joint_prob,
            'payout_multiplier': payout_multiplier,
            'expected_value': expected_value
        }

    def optimize_parlays(self, daily_props: list, min_legs=2, max_legs=8, top_n=10, beam_width=150) -> list:
        """
        Optimizes parlays using a Greedy Beam Search algorithm sorted by Expected Value (EV).
        """
        logger.info(f"Optimizing parlays for {len(daily_props)} props using EV Maximization...")
        
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
                # Keep ticket if the joint probability is valid
                if ticket_eval['joint_prob'] > 0:
                    evaluated_tickets.append(ticket_eval)
            
            if not evaluated_tickets:
                break

            # CRITICAL FIX: Sort strictly by Highest Expected Value Return
            leg_best = sorted(evaluated_tickets, key=lambda x: x['expected_value'], reverse=True)[:top_n]
            final_best_tickets.extend(leg_best)
            
            # Feed the highest EV tickets to the next beam level
            evaluated_tickets = sorted(evaluated_tickets, key=lambda x: x['expected_value'], reverse=True)[:beam_width]
            current_beams = [t['ticket'] for t in evaluated_tickets]

        return final_best_tickets