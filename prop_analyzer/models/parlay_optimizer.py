import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import norm
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

logger = logging.getLogger(__name__)

# Standard DFS / Sportsbook Payout Multipliers (Underdog Fantasy / PrizePicks standard)
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
    def __init__(self, historical_data: pd.DataFrame = None, num_simulations: int = 10000):
        """
        Initializes the Parlay Optimizer using Empirical Covariance mapping and Gaussian Copulas.
        """
        self.num_simulations = num_simulations
        self._simulation_cache = {}
        self.historical_logs = pd.DataFrame()
        logger.info("Initialized ParlayOptimizer with Empirical Covariance Mapping.")
        self._build_empirical_matrix(historical_data)

    def _build_empirical_matrix(self, df):
        """Loads historical box scores to calculate true pairwise correlations dynamically."""
        if df is None or df.empty:
            try:
                # Load the most recent master box score file to build the covariance matrix
                box_files = sorted(list(cfg.DATA_DIR.glob("master_box_scores_*.parquet")))
                if box_files:
                    df = pd.read_parquet(box_files[-1])
            except Exception as e:
                logger.warning(f"Could not load box scores for empirical covariance: {e}")
                return

        if df is None or df.empty: 
            return

        keep_cols = [Cols.PLAYER_NAME, Cols.TEAM, Cols.GAME_ID, 'PTS', 'REB', 'AST', 'MIN']
        avail_cols = [c for c in keep_cols if c in df.columns]
        
        if len(avail_cols) >= 4:
            # Filter out games where players barely played to avoid skewed correlation noise
            if 'MIN' in df.columns:
                df = df[df['MIN'] >= 10.0]
                
            self.historical_logs = df[avail_cols].dropna()
            logger.info(f"Successfully loaded {len(self.historical_logs)} historical logs for empirical correlation.")

    def get_correlation(self, prop1: dict, prop2: dict) -> float:
        """
        Determines the empirical correlation between two props in the same game based on their shared history.
        Uses exact historical covariance if sample size is large enough.
        """
        if prop1.get('game_id', prop1.get('Matchup')) != prop2.get('game_id', prop2.get('Matchup')):
            return 0.0

        p1_name = prop1.get('player_name', prop1.get(Cols.PLAYER_NAME, ''))
        p2_name = prop2.get('player_name', prop2.get(Cols.PLAYER_NAME, ''))
        
        # Base stat mapping to handle combo props cleanly in the matrix
        def get_base_stat(stat):
            stat = str(stat).upper()
            if stat in ['PRA', 'PR', 'PA', 'PTS']: return 'PTS'
            if stat in ['RA', 'REB']: return 'REB'
            if stat in ['AST']: return 'AST'
            return 'PTS' # Fallback
            
        base1 = get_base_stat(prop1.get('stat_type', prop1.get('PROP_TYPE', 'PTS')))
        base2 = get_base_stat(prop2.get('stat_type', prop2.get('PROP_TYPE', 'PTS')))
        
        # Calculate true empirical correlation on the fly using historical overlapping games
        if not self.historical_logs.empty and p1_name and p2_name:
            logs = self.historical_logs
            p1_games = logs[logs[Cols.PLAYER_NAME] == p1_name][['GAME_ID', base1]].set_index('GAME_ID')
            p2_games = logs[logs[Cols.PLAYER_NAME] == p2_name][['GAME_ID', base2]].set_index('GAME_ID')

            joint = p1_games.join(p2_games, how='inner', lsuffix='_1', rsuffix='_2')
            if len(joint) >= 15: # Need a minimum 15-game sample size to trust the mathematical correlation
                corr = joint[f"{base1}_1"].corr(joint[f"{base2}_2"])
                if not pd.isna(corr):
                    # Cap extreme correlations to prevent optimizer exploitation
                    return float(np.clip(corr, -0.45, 0.45))

        # Fallback to weak structural baselines ONLY if no history exists (e.g. recent trades, rookies)
        is_same_team = prop1.get('team', prop1.get(Cols.TEAM)) == prop2.get('team', prop2.get(Cols.TEAM))
        if is_same_team:
            if (base1 == 'PTS' and base2 == 'AST') or (base1 == 'AST' and base2 == 'PTS'): return 0.15
            if base1 == 'REB' and base2 == 'REB': return -0.15
            if base1 == 'PTS' and base2 == 'PTS': return -0.10
            if base1 == 'AST' and base2 == 'AST': return -0.20
        else:
            if base1 == 'PTS' and base2 == 'PTS': return 0.15
            if base1 == 'REB' and base2 == 'REB': return -0.15
            if base1 == 'AST' and base2 == 'AST': return 0.10

        return 0.0

    def simulate_same_game_cluster(self, cluster_props: list) -> float:
        """
        Uses a Gaussian Copula (Monte Carlo) to determine the true joint probability of a Same-Game Parlay.
        Adjusts expected distributions dynamically based on Game Script/Blowout risk.
        """
        n = len(cluster_props)
        if n == 1:
            return cluster_props[0].get('win_prob', cluster_props[0].get('Prob', 0.5))
            
        cache_key = frozenset([
            f"{p.get('player_name', p.get(Cols.PLAYER_NAME, ''))}_{p.get('stat_type', p.get('PROP_TYPE', ''))}_{p.get('pick', p.get('Pick', ''))}" 
            for p in cluster_props
        ])
        
        if cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]

        cov_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.get_correlation(cluster_props[i], cluster_props[j])
                
                # Invert correlation if taking opposite sides of the prop (e.g., Over/Under)
                pick_i = cluster_props[i].get('pick', cluster_props[i].get('Pick', 'Over'))
                pick_j = cluster_props[j].get('pick', cluster_props[j].get('Pick', 'Over'))
                if pick_i != pick_j:
                    corr = -corr
                    
                cov_matrix[i, j] = corr
                cov_matrix[j, i] = corr
                
        # PSD Check - Ensure matrix is positive semi-definite (required for multivariate_normal)
        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        while min_eig < 0:
            cov_matrix = (cov_matrix * 0.8) + (np.eye(n) * 0.2)
            min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
                
        # Convert standalone probabilities into Standard Normal thresholds
        thresholds = [norm.ppf(1 - prop.get('win_prob', prop.get('Prob', 0.5))) for prop in cluster_props]
        
        # BLOWOUT RISK COPULA ADJUSTMENT: 
        # High blowout risk mathematically shifts the mean distribution of the game, crushing overs for starters.
        mean = np.zeros(n)
        blowout_risk = max([prop.get('BLOWOUT_POTENTIAL', 0.0) for prop in cluster_props] + [0.0])
        
        if blowout_risk > 15.0:
            for i, prop in enumerate(cluster_props):
                # A 20.0 blowout risk shifts standard normal mean down by 0.2 standard deviations
                shift = min(blowout_risk / 100.0, 0.40) 
                pick = prop.get('pick', prop.get('Pick', 'Over'))
                is_bench = prop.get('IS_BENCH_ROLE', 0)
                
                # Blowouts hurt Starter Overs and help Starter Unders. 
                # Bench players might actually benefit from garbage time.
                if is_bench == 0: 
                    mean[i] = -shift if pick == 'Over' else shift
                else:
                    mean[i] = shift if pick == 'Over' else -shift

        try:
            samples = np.random.multivariate_normal(mean, cov_matrix, self.num_simulations)
            hits = np.all(samples > thresholds, axis=1)
            joint_prob = np.sum(hits) / self.num_simulations
        except np.linalg.LinAlgError:
            # Fallback to independent probability multiplication if matrix math completely fails
            logger.warning("LinAlgError in Copula simulation. Falling back to independent probability.")
            joint_prob = np.prod([prop.get('win_prob', prop.get('Prob', 0.5)) for prop in cluster_props])
        
        self._simulation_cache[cache_key] = joint_prob
        return joint_prob

    def calculate_ticket_metrics(self, ticket: list) -> dict:
        """
        Evaluates a full ticket (which may contain multiple games).
        Breaks the ticket down into Same-Game clusters for Copula simulation, 
        then multiplies the independent game clusters together.
        """
        num_legs = len(ticket)
        
        # Block single-team parlays if the platform requires multiple teams
        unique_teams = {prop.get('team', prop.get(Cols.TEAM)) for prop in ticket}
        if len(unique_teams) < 2:
            return {
                'ticket': ticket, 'legs': num_legs, 'joint_prob': 0.0,
                'payout_multiplier': 0.0, 'expected_value': 0.0
            }

        # Group props by game
        games = {}
        for prop in ticket:
            game_id = prop.get('game_id', prop.get('Matchup', 'unknown'))
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(prop)
            
        # Calculate joint probability cluster by cluster
        total_joint_prob = 1.0
        for game_id, cluster in games.items():
            cluster_prob = self.simulate_same_game_cluster(cluster)
            total_joint_prob *= cluster_prob
            
        payout_multiplier = UNDERDOG_PAYOUTS.get(num_legs, 0.0)
        expected_value = total_joint_prob * payout_multiplier
        
        return {
            'ticket': ticket, 'legs': num_legs, 'joint_prob': total_joint_prob,
            'payout_multiplier': payout_multiplier, 'expected_value': expected_value
        }

    def optimize_parlays(self, daily_props: list, top_n=20) -> list:
        """
        Takes the day's predicted props, filters for high confidence, 
        and iterates to find the absolute mathematically optimal parlays.
        """
        logger.info(f"Optimizing 2-leg parlays for {len(daily_props)} props...")
        viable_props = []
        
        # Initial Filtering: Only keep props with strong foundational probability and no Trap warnings
        for p in daily_props:
            prob = p.get('win_prob', p.get('Prob', 0))
            tier = p.get('Tier', 'Pass')
            
            # Reject high variance/trap plays entirely from the optimizer
            if tier not in ['Trap / High Variance', 'Pass / Too Volatile', 'Pass', 'Trap / Fade']:
                if prob >= 0.575: 
                    viable_props.append(p)
                
        # Limit the search space to the top 35 plays of the day to prevent combinatoric explosion
        viable_props = sorted(viable_props, key=lambda x: x.get('win_prob', x.get('Prob', 0)), reverse=True)[:35]
        
        if len(viable_props) < 2:
            logger.warning("Not enough viable props to form high-probability parlays today.")
            return []

        evaluated_tickets = []
        
        # Iterate through every possible 2-leg combination
        for prop1, prop2 in combinations(viable_props, 2):
            # Most DFS platforms restrict same-team only parlays. Skip them.
            if prop1.get('team', prop1.get(Cols.TEAM)) == prop2.get('team', prop2.get(Cols.TEAM)):
                continue 
            
            ticket = sorted([prop1, prop2], key=lambda x: x.get('player_name', x.get(Cols.PLAYER_NAME, '')))
            ticket_eval = self.calculate_ticket_metrics(ticket)
            
            # Only append tickets that passed valid team/game structural checks
            if ticket_eval['joint_prob'] > 0:
                evaluated_tickets.append(ticket_eval)

        # Sort all evaluated tickets by true joint probability of hitting
        final_best_tickets = sorted(evaluated_tickets, key=lambda x: x['joint_prob'], reverse=True)[:top_n]
        
        logger.info(f"Generated top {len(final_best_tickets)} optimal parlays.")
        return final_best_tickets