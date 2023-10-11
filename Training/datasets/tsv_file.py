import logging
import gc
import os
import os.path as op
from typing import List

logger = logging.getLogger(__name__)


def generate_lineidx(filein: str, idxout: str) -> None:
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class TSVFile(object):
    def __init__(self,
                 tsv_file: str,
                 if_generate_lineidx: bool = False,
                 lineidx: str = None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' \
            if not lineidx else lineidx
        self.linelist = op.splitext(tsv_file)[0] + '.linelist'
        self._fp = None
        self._lineidx = None
        self._sample_indices = None
        self._class_boundaries = None
        self._len = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and if_generate_lineidx:
            generate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        self.gcidx()
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def gcidx(self):
        logger.debug('Run gc collect')
        self._lineidx = None
        self._sample_indices = None
        #self._class_boundaries = None
        return gc.collect()

    def num_rows(self, gcf=False):
        if (self._len is None):
            self._ensure_lineidx_loaded()
            retval = len(self._sample_indices)

            if (gcf):
                self.gcidx()

            self._len = retval

        return self._len

    def seek(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[self._sample_indices[idx]]
        except:
            logger.info('=> {}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx: int):
        return self.seek_first_column(idx)

    def __getitem__(self, index: int):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logger.debug('=> loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                try:
                    lines = fp.readlines()
                except OSError:
                    logger.info(self.lineidx)
                    raise OSError
                lines = [line.strip() for line in lines]
                self._lineidx = [int(line) for line in lines]

            # read the line list if exists
            linelist = None
            if op.isfile(self.linelist):
                with open(self.linelist, 'r') as fp:
                    linelist = sorted(
                        [
                            int(line.strip())
                            for line in fp.readlines()
                        ]
                    )

            if linelist:
                self._sample_indices = linelist
            else:
                self._sample_indices = list(range(len(self._lineidx)))

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logger.debug('=> re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class CompositeTSVFile:
    def __init__(self,
                 file_list: List[str],
                 root: str = '.'):
        self.root = root
        self.tsvs = None
        self.chunk_sizes = None
        self.accum_chunk_sizes = None
        self.initialized = False
        assert isinstance(file_list, list)
        self.file_list = file_list
        self.initialize()

    def get_key(self, index: int):
        idx_source, idx_row = self._calc_chunk_idx_row(index)
        k = self.tsvs[idx_source].get_key(idx_row)
        return '_'.join([self.file_list[idx_source], k])

    def get_chunk_size(self):
        return self.chunk_sizes

    def num_rows(self):
        return sum(self.chunk_sizes)

    def _calc_chunk_idx_row(self, index: int):
        idx_chunk = 0
        idx_row = index
        while index >= self.accum_chunk_sizes[idx_chunk]:
            idx_chunk += 1
            idx_row = index - self.accum_chunk_sizes[idx_chunk-1]
        return idx_chunk, idx_row

    def __getitem__(self, index: int):
        idx_source, idx_row = self._calc_chunk_idx_row(index)
        idx_source_file = TSVFile(
            op.join(self.root, self.file_list[idx_source]),
        )
        return idx_source_file.seek(idx_row)

    def __len__(self):
        return sum(self.chunk_sizes)

    def initialize(self):
        """
        this function has to be called in init function if cache_policy is
        enabled. Thus, let's always call it in init funciton to make it simple.
        """
        if self.initialized:
            return
        self.tsvs = [
            TSVFile(
                op.join(self.root, f),
            ) for f in self.file_list
        ]
        logger.debug("=> Calculating chunk sizes ...")
        self.chunk_sizes = [tsv.num_rows(gcf=True) for tsv in self.tsvs]

        self.accum_chunk_sizes = [0]
        for size in self.chunk_sizes:
            self.accum_chunk_sizes += [self.accum_chunk_sizes[-1] + size]
        self.accum_chunk_sizes = self.accum_chunk_sizes[1:]
        self.initialized = True


def load_list_file(fname: str) -> List[str]:
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result