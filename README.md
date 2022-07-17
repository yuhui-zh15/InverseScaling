# Inverse Scaling


## 07/17

- [ ] Setup [OPT](https://github.com/facebookresearch/metaseq)
- [ ] Implement an interface `get_language_modeling_scores(input_str: str) -> List[Tuple(str, float)]`, each output tuple is token string and language model output log probablitity. (e.g., `get_language_modeling_scores("hello world") -> [("hello", -3.428), ("world", -1.203)]`)
