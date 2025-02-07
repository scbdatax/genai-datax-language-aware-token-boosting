from typing import List, Dict

from scipy.stats import ttest_ind, f_oneway


class HypothesisTestingDirection:
    LESS = "less"
    GREATER = "greater"
    TWO_TAILED = "two-sided"
    

def perform_hypothesis_testing(
    a: List[int], 
    b: List[int], 
    *args, 
    direction: HypothesisTestingDirection = HypothesisTestingDirection.LESS,
    random_state: int = 123
    ) -> Dict:
    if args:
        # Perform ANOVA
        all_groups = [a, b, *args]
        anova_result = f_oneway(*all_groups)
        return {
            "test": "ANOVA",
            "p_value": anova_result.pvalue,
            "statistic": anova_result.statistic
        }
    else:
        # Perform t-test
        t_test_result = ttest_ind(a, b, equal_var=False, alternative=direction, random_state=random_state)
        return {
            "test": "t-test",
            "p_value": t_test_result.pvalue,
            "statistic": t_test_result.statistic
        }
