import pandas as pd
from typing import Dict, FrozenSet, List, Tuple


def _intersect_tids(tids1, tids2):
    # tids1, tids2 are Python sets; intersection is fast
    return tids1 & tids2


def eclat(
    tidlists: Dict[FrozenSet[str], set],
    min_support_count: int,
    prefix: FrozenSet[str] = frozenset(),
) -> Dict[FrozenSet[str], int]:
    """
    Very simple recursive Eclat algorithm.
    tidlists: mapping {frozenset({item}): set(transaction_ids)}
    min_support_count: minimum number of transactions
    prefix: current itemset prefix (for recursion)

    Returns:
      {itemset: support_count}
    """
    freq_itemsets: Dict[FrozenSet[str], int] = {}

    items = list(tidlists.items())
    for i, (itemset_i, tids_i) in enumerate(items):
        # Build new itemset by extending prefix
        new_itemset = prefix | itemset_i
        support = len(tids_i)
        if support >= min_support_count:
            freq_itemsets[new_itemset] = support

            # Build conditional tidlists for further extensions
            suffix_tidlists: Dict[FrozenSet[str], set] = {}
            for itemset_j, tids_j in items[i + 1 :]:
                inter = _intersect_tids(tids_i, tids_j)
                if len(inter) >= min_support_count:
                    suffix_tidlists[itemset_j] = inter

            if suffix_tidlists:
                # Recurse with new prefix and conditional tidlists
                deeper = eclat(suffix_tidlists, min_support_count, new_itemset)
                freq_itemsets.update(deeper)

    return freq_itemsets


def eclat_from_basket(
    basket: pd.DataFrame,
    min_support: float = 0.01,
) -> List[Tuple[FrozenSet[str], float, int]]:
    """
    Convenience function to run Eclat on a one-hot basket DataFrame.
    basket: DataFrame of shape (n_transactions, n_items), values are boolean
    """
    n_transactions = basket.shape[0]
    min_support_count = max(1, int(min_support * n_transactions))

    #  Build initial TID-lists for single items
    tidlists: Dict[FrozenSet[str], set] = {}
    for col in basket.columns:
        tids = set(basket.index[basket[col]])
        if len(tids) >= min_support_count:
            tidlists[frozenset({col})] = tids

    # Run recursive Eclat
    freq_itemsets_counts = eclat(tidlists, min_support_count)

    #  Convert to list with relative support
    results: List[Tuple[FrozenSet[str], float, int]] = []
    for itemset, count in freq_itemsets_counts.items():
        support = count / n_transactions
        results.append((itemset, support, count))

    return results