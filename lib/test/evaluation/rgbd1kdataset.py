import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class RGBD1KDataset(BaseDataset):
    """ RGBD1K dataset
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbd1k_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info
        nz = 8
        start_frame = 1
        ext = ['jpg', 'png']
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_path)
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        end_frame = ground_truth_rect.shape[0]

        color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                                                                                     sequence_path=sequence_path,
                                                                                     frame=frame_num, nz=nz)
                        for frame_num in range(start_frame, end_frame + 1)]

        depth_frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]

        frames = []
        for c_path, d_path in zip(color_frames, depth_frames):
            frames.append({'color': c_path, 'depth': d_path})

        return Sequence(sequence_info, frames, 'rgbd', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        list_file = os.path.join(self.base_path, 'list.txt')
        with open(list_file, 'r') as f:
            sequence_list = f.read().splitlines()
        return sequence_list
