# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] thread-pool batch loader (no spawned dataloader processes -> runs in the 'light' GPU slot; the global
heavy-training cap counts spawn-worker trainings). Same sampling as DataLoader(shuffle=True, drop_last=False, batch_size=B): a seeded random
permutation per epoch, samples fetched in parallel threads (JPEG decode / cv2 / numpy release the GIL), same collate_fn, one batch prefetched.
Same idea as GUPNet uda_train.py ThreadBatchLoader (workers 0 + threads)."""
import threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch


class ThreadLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, threads=6, seed=0, prefetch=3):
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last
        self.collate_fn = dataset.collate_batch
        self.pool = ThreadPoolExecutor(threads)
        self.g = torch.Generator(); self.g.manual_seed(int(seed))
        self.prefetch = prefetch

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def _batches(self):
        n = len(self.dataset)
        order = torch.randperm(n, generator=self.g).tolist() if self.shuffle else list(range(n))
        for k in range(len(self)):
            yield order[k * self.batch_size:(k + 1) * self.batch_size]

    def __iter__(self):
        q = queue.Queue(self.prefetch); stop = object()

        def prod():
            try:
                for ids in self._batches():
                    q.put(self.collate_fn(list(self.pool.map(self.dataset.__getitem__, ids))))
            except BaseException as e:   # surface errors in the consumer
                q.put(e)
            q.put(stop)
        threading.Thread(target=prod, daemon=True).start()
        while True:
            b = q.get()
            if b is stop:
                return
            if isinstance(b, BaseException):
                raise b
            yield b
