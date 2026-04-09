from typing import Dict, List

UNKNOWN_ITEM_INDEX = 0
UNKNOWN_ITEM_ID = "unknown"
MASK_ITEM_ID = "mask"
QUERY_ITEM_ID = "query"

SPECIAL_TOKENS = [UNKNOWN_ITEM_ID, MASK_ITEM_ID, QUERY_ITEM_ID]

PAD_ITEM_INDEX = -1
TOKENIZER_FILE = "ably_goods.model"
WORD2VEC_FILE = "word2vec.wordvectors"

EVENT_VOCA = {
    "unknown": 0,
    "click": 1,
    "preference": 2,
    "query": 3,
    "search_click": 4,
    "search_preference": 5,
}

EVENT_MAP = {
    "click": "click",
    "like": "preference",
    "cart": "preference",
    "order": "preference",
    "query": "query",
    "view": "click",
    "purchase": "preference",
    "deeplink": "preference",
}

COMPLEMENT_EVENT_MAP = {
    "click": "click",
    "like": "like",
    "cart": "cart",
    "order": "order",
    "query": "query",
    "view": "click",
    "purchase": "order",
}

CKPT_FILENAME = "best_acc_model.ckpt"
TRAINED_MODEL_FILENAME = "model.tar.gz"
TRAINED_CONFIG_FILENAME = "config.json"

STATE_DICT_KEY = "model_state_dict"
OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"

TEXT_MODEL_NAME = "distiluse-base-multilingual-cased"
TEXT_EMBEDDING_SIZE = 512

DF_ITEMS_FILENAME = "df_items.pkl"

USER2ITEMS_PROCESSED_FILE = "df_user2items.pkl"
ON_VACA_ITEMS_PROCESSED_FILE = "on_voca_items.pkl"


class RawDataField(object):
    ITEM_ID = "ITEM_ID"
    USER_ID = "USER_ID"
    TIMESTAMP = "TIMESTAMP"
    EVENT_TYPE = "EVENT_TYPE"
    MARKET_ID = "market_sno"
    MARKET_TYPE_ID = "market_type_sno"
    CATEGORY_ID = "category_sno"
    STANDARD_CATEGORY_ID = "standard_category_sno"
    PARENT_CATEGORY_ID = "parent_category__sno"
    PARENT_STANDARD_CATEGORY_ID = "parent_standard_category__sno"
    ITEM_PRICE = "price"
    TASTE_ID = "TASTE_ID"
    IS_SEARCH = "is_search"
    SCREEN_NAME = "screen_name"
    EVENT_NAME = "event_name"
    PASTEL_CATEGORY_L1 = "CATEGORY_L1"
    POSITIVE_REVIEW_COUNT = "positive_review_count"
    TOTAL_REVIEW_COUNT = "total_review_count"
    PURCHASE = "purchase"
    LIKE = "like"
    CART = "cart"


class DatasetField(object):
    ITEM_INDEX = "item_index"
    TIME_INTERVAL = "time_interval"
    EVENT_CODE = "EVENT_CODE"

    ITEM_TEXT = "text"
    MARKET_INDEX = "market_index"
    MARKET_TYPE_INDEX = "market_type_index"
    CATEGORY_INDEX = "category_index"
    STANDARD_CATEGORY_INDEX = "standard_category_index"
    ITEM_PRICE = "price"

    USER_AGE = "age"


class FeatureField(object):
    # User features
    UNCLICK_ITEMS = "unclick_items"
    CLICK_ITEMS = "click_items"
    LIKE_ITEMS = "like_items"
    USER_AGE = "user_age"
    USER_CODE = "user_code"
    CLICK_MARKETS = "click_markets"
    CLICK_MARKET_TYPES = "click_market_types"
    CLICK_CATEGORIES = "click_categories"
    CLICK_STANDARD_CATEGORIES = "click_standard_categories"
    CLICK_PRICE_GROUPS = "click_price_groups"
    SEARCH_QUERY = "search_query"
    SEARCH_QUERIES = "search_queries"

    # Item features
    ITEM = "item"
    ITEM_DROPOUT = "item_dropout"

    ITEM_MARKET = "item_market"
    ITEM_PRICE = "item_price"
    ITEM_TAG = "item_tag"
    ITEM_CATEGORY = "item_category"
    ITEM_STANDARD_CATEGORY = "item_standard_category"
    ITEM_MARKET_TYPE = "item_market_type"
    ITEM_TEXT_TOKEN = "item_text_token"

    ITEM_CTR_SCORE = "item_ctr_score"
    ITEM_CLICK_SCORE = "item_click_score"
    ITEM_REVIEW_SCORE = "item_review_score"
    ITEM_POS_REVIEW_SCORE = "item_pos_review_score"
    ITEM_PRICE_SCORE = "item_price_score"
    ITEM_PRICE_GROUP = "item_price_group"
    ITEM_DISPLAY_SCORE = "item_display_score"

    EVENT_CODE = "event_code"
    TIME_INTERVAL = "time_interval"

    @classmethod
    def name(cls):
        return "features"

    @classmethod
    def use_time_info(cls, feature_names: List):
        if FeatureField.TIME_INTERVAL in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_age_info(cls, feature_names: List):
        if FeatureField.USER_AGE in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_event_code(cls, feature_names: List):
        if FeatureField.LIKE_ITEMS in feature_names or FeatureField.EVENT_CODE in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_item_market_info(cls, feature_names: List):
        if FeatureField.ITEM_MARKET in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_item_category_info(cls, feature_names: List):
        if FeatureField.ITEM_CATEGORY in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_item_standard_category_info(cls, feature_names: List):
        if FeatureField.ITEM_STANDARD_CATEGORY in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_item_price_info(cls, feature_names: List):
        if FeatureField.ITEM_PRICE in feature_names:
            return True
        else:
            return False

    @classmethod
    def use_item_market_type_info(cls, feature_names: List):
        if FeatureField.ITEM_MARKET_TYPE in feature_names:
            return True
        else:
            return False

    @classmethod
    def get_feature_fields(cls, feature_process_spec: Dict) -> List[str]:
        fields = []
        for layer_name, spec in feature_process_spec.items():
            fields += spec[FeatureField.name()]
        return list(set(fields))
