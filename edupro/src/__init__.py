
from .utils import (
    CATEGORIES, LEVELS, COURSE_TYPES, GENDERS,
    SEGMENT_NAMES, SEGMENT_COLORS, SEGMENT_DESC,
    CLUSTERING_FEATURES, N_USERS, N_COURSES, N_CLUSTERS, RANDOM_SEED,
    SCORE_WEIGHTS, depth_index_to_level, adjacent_level
)
from .data_pipeline import generate_data, load_raw_data, save_raw_data
from .features import engineer_features
from .clustering import run_clustering, get_cluster_profiles, attach_cluster_labels
from .recommender import recommend_courses, recommend_new_user, evaluate_recommendation_quality

__all__ = [
    'CATEGORIES', 'LEVELS', 'COURSE_TYPES', 'GENDERS',
    'SEGMENT_NAMES', 'SEGMENT_COLORS', 'SEGMENT_DESC',
    'CLUSTERING_FEATURES', 'N_USERS', 'N_COURSES', 'N_CLUSTERS', 'RANDOM_SEED',
    'SCORE_WEIGHTS', 'depth_index_to_level', 'adjacent_level',
    'generate_data', 'load_raw_data', 'save_raw_data',
    'engineer_features',
    'run_clustering', 'get_cluster_profiles', 'attach_cluster_labels',
    'recommend_courses', 'recommend_new_user', 'evaluate_recommendation_quality',
]   

from .feedback import (
    save_feedback, load_feedback,
    get_feedback_stats, get_liked_courses,
    get_disliked_courses, compute_feedback_weights
)


from .auth import authenticate, get_allowed_pages, add_user, change_password, get_all_users, load_login_log
