import datasets
import datasets.turtle
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class turtle(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'turtle')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png', '.JPEG', '.JPG']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.roidb_builder

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 800}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def _load_image_set_index(self):
        """
        Load the indexes (image names) from the descriptor txt file
        """
        image_set_file = os.path.join(self._data_path, self._image_set+'.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        The index is the filename without extension.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'positives',
                                  index + ext)
            if os.path.exists(image_path):
                break

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def roidb_builder(self):
        """
        Return the database of regions of interest.
        Region of interest are computed with Multibox.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_multibox_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_multibox_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_multibox_roidb(None)
            print len(roidb)

        with open(cache_file, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)

        print 'wrote ss roidb to {}'.format(cache_file)
        return roidb

    def gt_roidb(self):
        """
        Read the GT regions from annotations
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_turtle_annotations()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_multibox_roidb(self, gt_roidb):
        # load multibox file
        # respectively train_multibox.pkl and test_multibox.pkl
        filename = os.path.abspath(os.path.join(self._data_path,
                                                self.name + '_multibox.pkl'))
        assert os.path.exists(filename), \
               'Multibox data not found at: {}'.format(filename)
        with open(filename, 'rb') as fid:
            raw_data = cPickle.load(fid)

        # ATTENZIONE: CHECK ORDINE COORDINATE
        # ORA x1 y1 x2 y2
        # box_list = []
        # for i in xrange(raw_data.shape[0]):
        #         raw_data[i] = raw_data[i][:, (1, 0, 3, 2)]

        return self.create_roidb_from_box_list(raw_data, gt_roidb)

    def _load_turtle_annotations(self):
        """
        Read annotations file and return a list of dictionaries.
        Each element has the form
        {'boxes' : box,
        'gt_classes': gt_class,
        'gt_overlaps' : 1,
        'flipped' : False}
        """
        ret = []
        num_objs = 1
        # open annotations file
        with open(os.path.join(self._data_path, 'bboxs.txt')) as f:
            lines = [l.strip().split(' ') for l in f.readlines()]
            bboxs = {l[0] : l[1:] for l in lines}

        for index in self.image_index:
            # initialize annotations
            box = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_class = np.zeros((num_objs), dtype=np.uint16)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # get full path
            img = self.image_path_from_index(index)
            # search full path in annotations
            bbox = [int(c) for c in bboxs[img]]
            # fill annotations
            box[0, :] = np.array(bbox)
            cls_ = self._class_to_ind['turtle']
            gt_class[0] = cls_
            overlaps[0, cls_] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)
            ret.append(
                {'boxes' : box,
                'gt_classes': gt_class,
                'gt_overlaps' : overlaps,
                'flipped' : False}
            )

        return ret


if __name__ == '__main__':
    d = datasets.turtle('train', '')
    res = d.roidb
    from IPython import embed
    embed()
