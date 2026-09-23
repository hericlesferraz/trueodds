import random

from trueodds.batching import padded_tokens, plan_epoch, split_microbatches


def data(n=500, seed=0):
    rng = random.Random(seed)
    lengths = [rng.randint(10, 512) for _ in range(n)]
    ks = [rng.choice([2, 3, 4, 10]) for _ in range(n)]
    return lengths, ks


def test_every_question_once_per_epoch_in_steps_of_at_most_32():
    lengths, ks = data()
    steps = plan_epoch(lengths, ks, 32, random.Random(0), budget=16_384)
    flat = [i for s in steps for i in s]
    assert sorted(flat) == list(range(len(lengths)))
    assert all(len(s) <= 32 for s in steps)
    assert sum(len(s) == 32 for s in steps) >= len(steps) - 8  # only chunk ends are short


def test_steps_group_similar_option_counts():
    lengths, ks = data(n=2048)
    steps = plan_epoch(lengths, ks, 32, random.Random(0), budget=16_384)
    one_k = sum(len({ks[i] for i in s}) == 1 for s in steps)
    assert one_k / len(steps) > 0.9


def test_microbatches_stay_under_the_budget():
    lengths, ks = data()
    for step in plan_epoch(lengths, ks, 32, random.Random(0), budget=16_384):
        ls, kk = [lengths[i] for i in step], [ks[i] for i in step]
        mbs = split_microbatches(ls, kk, 16_384)
        assert sorted(i for mb in mbs for i in mb) == list(range(len(step)))
        for mb in mbs:
            cost = padded_tokens([ls[i] for i in mb], [kk[i] for i in mb])
            assert cost <= 16_384 or len(mb) == 1


def test_a_question_over_the_budget_gets_its_own_microbatch():
    assert split_microbatches([512, 10], [14, 2], budget=4096) == [[0], [1]]


def test_longest_first_puts_the_costliest_step_first():
    lengths, ks = data()
    budget = 16_384
    steps = plan_epoch(lengths, ks, 32, random.Random(0), budget, longest_first=True)

    def cost(step):
        ls, kk = [lengths[i] for i in step], [ks[i] for i in step]
        return max(
            padded_tokens([ls[i] for i in mb], [kk[i] for i in mb])
            for mb in split_microbatches(ls, kk, budget)
        )

    costs = [cost(s) for s in steps]
    assert costs == sorted(costs, reverse=True)
