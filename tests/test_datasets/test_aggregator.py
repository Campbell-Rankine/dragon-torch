import pytest
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
from dragon_torch.data.datasets.aggregator import Aggregator


class ListDataset:
    """Minimal map-style dataset: sample i is (name, i)."""

    def __init__(self, name: str, length: int, kind: str = None):
        self.name = name
        self.length = length
        self.kind = kind if kind is not None else name

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if not 0 <= i < self.length:
            raise IndexError(i)
        return (self.name, i)


@pytest.fixture
def a():
    return ListDataset('a', 20)


@pytest.fixture
def b():
    return ListDataset('b', 30)


def passthrough(batch):
    # module-level so spawned DataLoader workers can pickle it
    return batch


def flatten(batches, dataset_idx):
    return [s for batch in batches for s in batch[dataset_idx]]


class TestAggregatorInit:
    def test_is_iterable_dataset(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5})
        assert isinstance(agg, IterableDataset)

    @pytest.mark.parametrize('batch_size,weights,expected', [
        (4, {0: 0.5, 1: 0.5}, [2, 2]),
        (10, {0: 0.3, 1: 0.7}, [3, 7]),
        (5, {0: 0.5, 1: 0.5}, [2, 2]),  # floored, batch comes out short
        (4, {0: 1.0, 1: 0.0}, [4, 0]),
    ])
    def test_samples_per_dataset(self, a, b, batch_size, weights, expected):
        agg = Aggregator([a, b], batch_size=batch_size, weights=weights)
        assert agg.samples_per_dataset == expected

    def test_weights_by_kind(self, a, b):
        agg = Aggregator([a, b], batch_size=10, weights={'b': 0.6, 'a': 0.4})
        assert agg.samples_per_dataset == [4, 6]

    def test_weights_mixed_keys(self, a, b):
        agg = Aggregator([a, b], batch_size=10, weights={0: 0.4, 'b': 0.6})
        assert agg.samples_per_dataset == [4, 6]

    def test_does_not_mutate_inputs(self, a, b):
        datasets, weights = [a, b], {0: 0.5, 1: 0.5}
        Aggregator(datasets, batch_size=4, weights=weights)
        assert datasets == [a, b] and weights == {0: 0.5, 1: 0.5}

    def test_caches_lengths(self, a, b, monkeypatch):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5})
        monkeypatch.setattr(ListDataset, '__len__',
                            lambda self: pytest.fail('len re-queried'))
        len(agg)
        agg[0]

    @pytest.mark.parametrize('weights,error', [
        ({0: 1.0}, ValueError),                     # too few
        ({0: 0.5, 1: 0.5, 2: 0.0}, ValueError),     # too many
        ({0: 0.5, 2: 0.5}, KeyError),               # index out of range
        ({0: 0.5, 'c': 0.5}, KeyError),             # unknown kind
        ({0: 0.5, 'a': 0.5}, ValueError),           # dataset 0 weighted twice
        ({0: -0.1, 1: 0.5}, ValueError),            # negative
        ({0: 0.7, 1: 0.7}, ValueError),             # sums past 1
        ({0: 0.0, 1: 0.0}, ValueError),             # draws nothing
    ])
    def test_invalid_weights(self, a, b, weights, error):
        with pytest.raises(error):
            Aggregator([a, b], batch_size=4, weights=weights)

    def test_ambiguous_kind(self):
        x, y = ListDataset('x', 10, kind='same'), ListDataset('y', 10, kind='same')
        with pytest.raises(KeyError, match='same'):
            Aggregator([x, y], batch_size=4, weights={'same': 0.5, 1: 0.5})

    def test_small_weight_floors_to_zero(self, a, b):
        with pytest.raises(ValueError):
            Aggregator([a, b], batch_size=2, weights={0: 0.4, 1: 0.4})

    @pytest.mark.parametrize('batch_size', [0, -1])
    def test_invalid_batch_size(self, a, b, batch_size):
        with pytest.raises(ValueError):
            Aggregator([a, b], batch_size=batch_size, weights={0: 0.5, 1: 0.5})

    def test_no_datasets(self):
        with pytest.raises(ValueError):
            Aggregator([], batch_size=4, weights={})


class TestAggregatorLength:
    def test_limited_by_first_exhausted(self, a, b):
        # a: 20 // 2 = 10 batches, b: 30 // 2 = 15 batches
        assert len(Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5})) == 10

    def test_weight_shifts_limit(self, a, b):
        # a: 20 // 1 = 20 batches, b: 30 // 3 = 10 batches
        assert len(Aggregator([a, b], batch_size=4, weights={0: 0.25, 1: 0.75})) == 10

    def test_zero_weight_dataset_ignored(self, a):
        tiny = ListDataset('tiny', 1)
        assert len(Aggregator([a, tiny], batch_size=4, weights={0: 1.0, 1: 0.0})) == 5

    def test_dataset_smaller_than_share(self, a):
        tiny = ListDataset('tiny', 1)
        assert len(Aggregator([a, tiny], batch_size=4, weights={0: 0.5, 1: 0.5})) == 0


class TestAggregatorGetItem:
    def test_batch_layout(self, a, b):
        agg = Aggregator([a, b], batch_size=10, weights={0: 0.3, 1: 0.7}, shuffle=False)
        batch = agg[0]
        assert len(batch) == 2
        assert [len(part) for part in batch] == [3, 7]
        assert all(name == 'a' for name, _ in batch[0])
        assert all(name == 'b' for name, _ in batch[1])

    def test_unshuffled_is_sequential(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, shuffle=False)
        assert agg[0] == [[('a', 0), ('a', 1)], [('b', 0), ('b', 1)]]
        assert agg[3] == [[('a', 6), ('a', 7)], [('b', 6), ('b', 7)]]

    def test_zero_weight_gives_empty_part(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 1.0, 1: 0.0})
        assert agg[0][1] == []

    def test_negative_index(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        assert agg[-1] == agg[len(agg) - 1]

    @pytest.mark.parametrize('idx', [10, 11, -11])
    def test_out_of_range(self, a, b, idx):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5})
        with pytest.raises(IndexError):
            agg[idx]

    def test_repeatable_within_epoch(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5})
        assert agg[2] == agg[2]

    def test_collate_fn(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, shuffle=False,
                         collate_fn=lambda batch: [s for part in batch for s in part])
        assert agg[0] == [('a', 0), ('a', 1), ('b', 0), ('b', 1)]

    def test_hf_dataset(self):
        x = Dataset.from_dict({'v': list(range(8))})
        y = Dataset.from_dict({'v': list(range(100, 108))})
        agg = Aggregator([x, y], batch_size=4, weights={0: 0.25, 1: 0.75}, shuffle=False)
        assert len(agg) == 2
        assert agg[0] == [[{'v': 0}], [{'v': 100}, {'v': 101}, {'v': 102}]]


class TestAggregatorShuffle:
    def test_epoch_draws_without_replacement(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        drawn_a, drawn_b = flatten(list(agg), 0), flatten(list(agg), 1)
        assert sorted(i for _, i in drawn_a) == list(range(20))
        assert len(set(drawn_b)) == len(drawn_b) == 20

    def test_shuffles(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        assert flatten(list(agg), 0) != [('a', i) for i in range(20)]

    def test_seed_is_deterministic(self, a, b):
        kw = dict(batch_size=4, weights={0: 0.5, 1: 0.5}, seed=7)
        assert list(Aggregator([a, b], **kw)) == list(Aggregator([a, b], **kw))

    def test_seeds_differ(self, a, b):
        kw = dict(batch_size=4, weights={0: 0.5, 1: 0.5})
        assert list(Aggregator([a, b], seed=1, **kw)) != list(Aggregator([a, b], seed=2, **kw))

    def test_set_epoch_reshuffles(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        first = list(agg)
        agg.set_epoch(1)
        second = list(agg)
        assert first != second
        assert sorted(flatten(first, 0)) == sorted(flatten(second, 0))

    def test_set_epoch_is_reproducible(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        first = list(agg)
        agg.set_epoch(1)
        agg.set_epoch(0)
        assert list(agg) == first


class TestAggregatorIteration:
    def test_iter_matches_getitem(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        assert list(agg) == [agg[i] for i in range(len(agg))]

    def test_iter_is_restartable(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        assert list(agg) == list(agg)

    def test_dataloader(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        loader = DataLoader(agg, batch_size=None, collate_fn=passthrough)
        assert list(loader) == list(agg)

    def test_dataloader_workers_split_batches(self, a, b):
        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0)
        loader = DataLoader(agg, batch_size=None, num_workers=2,
                            collate_fn=passthrough)
        batches = list(loader)
        assert len(batches) == len(agg)
        key = lambda batch: str(batch)
        assert sorted(batches, key=key) == sorted(list(agg), key=key)
