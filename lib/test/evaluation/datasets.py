from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    rgbd1k=DatasetInfo(module=pt % "rgbd1k", class_name="RGBD1KDataset", kwargs=dict()),
    cdtb=DatasetInfo(module=pt % "cdtb", class_name="CDTBDDataset", kwargs=dict()),
    depthtrack=DatasetInfo(module=pt % "depthtrack", class_name="DepthTrackDataset", kwargs=dict())
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
