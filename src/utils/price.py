import numpy as np
import pandas as pd
from utils.constants import FeatureField


def get_standard_category_price_info(df_items: pd.DataFrame) -> pd.DataFrame:
    """카테고리별 가격 정보를 구한다.
    구체적으로는, 가격의 분포에서 p33과 p67을 구해서 데이터프레임으로 리턴한다.
    """
    df_standard_category_price_percentile = (
        df_items.groupby(FeatureField.ITEM_STANDARD_CATEGORY)[FeatureField.ITEM_PRICE]
        .quantile([0.33, 0.67])
        .unstack()
        .reset_index()
    )
    df_standard_category_price_percentile.columns = [FeatureField.ITEM_STANDARD_CATEGORY, "p33", "p67"]
    return df_standard_category_price_percentile


def add_item_price_group_column(df_items, df_standard_category_price_percentile):
    """df_items에 상품의 price_group 컬럼을 추가된 데이터프레임을 리턴한다.
    price_group: 상품의 가격이 그 카테고리의 p33보다 작으면 1, p67보다 크면 3, 그 사이면 2.
    """
    return (
        pd.merge(df_items, df_standard_category_price_percentile, on=FeatureField.ITEM_STANDARD_CATEGORY, how='left')
        .fillna({'p33': 0, 'p67': 0})
        .assign(
            item_price_group=lambda x: np.where(x.item_price < x.p33, 1, np.where(x.item_price > x.p67, 3, 2))
        )
    )
