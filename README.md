# RecSysHM

Thêm Random Walk candidates.

## `create_random_walk_candidates`

Hàm `create_random_walk_candidates` (trong [candidates.py](candidates.py)) thêm một chiến lược sinh candidate dựa trên **random walk with restart (RWR)** trên đồ thị item-item.

### Ý tưởng

- Ta xem `article_pairs_df` như một đồ thị có hướng: `article_id -> pair_article_id`.
- Trọng số cạnh lấy từ `weight_col` (mặc định `customer_count`) và được **chuẩn hoá theo outgoing** để tạo phân phối chuyển trạng thái.
- Với mỗi customer, ta tạo phân phối seed $p_0$ trên các item họ vừa mua gần đây.
- Sau đó chạy random walk $K$ bước và ở mỗi bước trộn giữa:

$$
score_{t+1} = r\,p_0 + (1-r)\,propagate(score_t)
$$

Trong đó $r$ là `restart_prob` (cố định hoặc theo-customer khi dùng `"adaptive"`).

### Cách implement

1. Thêm hàm `create_random_walk_candidates` vào cuối [candidates.py]

2. Trong notebook phần creating candidates (and adding features) thêm
```
random_walk_cand, features_db["f_random_walk"] = h_can.create_random_walk_candidates(
    features_df,
    article_pairs_df,
    seed_weeks=kwargs["gd_seed_weeks"],
    seed_articles=kwargs["gd_seed_articles"],
    num_steps=kwargs["gd_steps"],
    restart_prob=kwargs["gd_restart_prob"],
    topk=kwargs["gd_topk"],
    weight_col=kwargs["gd_weight_col"],
    recency_weight=kwargs["gd_recency_weight"],
    exclude_seed_items=kwargs["gd_exclude_seed"],
    customers=customers,
)
```
3. Trong notebook phần creating candidates (and adding features) sửa
```
cand = [
    recent_customer_cand,
    cust_last_week_cand,
    cust_last_week_pair_cand,
    random_walk_cand, #NEW
    popular_cand,
    age_bucket_can,
]
```
```
del (
    recent_customer_cand,
    cust_last_week_cand,
    cust_last_week_pair_cand,
    random_walk_cand, #NEW
    age_bucket_can,
    popular_cand,
)
```
4. Trong notebook phần parameters thêm vào `cv_params` và `sub_params` các tham số sau
```
# random walk with restart candidate generation
"gd_seed_weeks": 12,
"gd_seed_articles": 12,
"gd_steps": 2,
"gd_restart_prob": "adaptive",
"gd_topk": 24,
"gd_weight_col": "customer_count",
"gd_recency_weight": True,
"gd_exclude_seed": True,