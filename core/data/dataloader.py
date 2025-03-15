import itertools
import multiprocessing as mp
import os
import pickle
import queue
import random
import threading
import time
import math
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import Dataset, _utils
from torch.utils.data.dataloader import (DataLoader, _DatasetKind,
                                         _MultiProcessingDataLoaderIter,
                                         _SingleProcessDataLoaderIter,
                                         _BaseDataLoaderIter,
                                         default_collate)  # IterDataPipe, MapDataPipe,
import rpyc     
from rpyc.utils.server import ThreadedServer, Server
from multiprocessing import Process
rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
 
rpyc_config['allow_pickle'] = True
rpyc_config['sync_request_timeout'] = 600

def get_address_ip():
    import socket
    return socket.gethostbyname(socket.getfqdn(socket.gethostname()))

def queue_actor_run(t):
    print("--queue_actor_run start--", t)
    t.start()
    print("--queue_actor_run end--", t)                                        

class Preprocess():

    def has_cpu_preprocess(self):
        return False
    
    def has_gpu_preprocess(self):
        return False

    def cpu_preprocess(self, data):
        return data
    
    def gpu_preprocess(self, data, stream):
        return data


class _Preprocess_DatasetFetcher():

    def __init__(self, dataset_fetcher, preprocess):
        self.dataset_fetcher = dataset_fetcher
        self.preprocess = preprocess
    
    def fetch(self, possibly_batched_index):
        # print(f"_Preprocess_DatasetFetcher fetch possibly_batched_index = {possibly_batched_index}")
        data = self.dataset_fetcher.fetch(possibly_batched_index)
        if self.preprocess is not None and self.preprocess.has_cpu_preprocess():
            data = self.preprocess.cpu_preprocess(data)
            data = _utils.collate.default_convert(data)
        return data

def _create_fetcher_proxy(create_fetcher_fn, preprocess):
    def wrapper(*args, **kws):
        # print("_create_fetcher_proxy wrapper")
        fetcher = create_fetcher_fn(*args, **kws)
        if preprocess is not None and preprocess.has_cpu_preprocess():
           fetcher = _Preprocess_DatasetFetcher(fetcher, preprocess)
        return fetcher
    return wrapper


class _PreprocessSingleProcessDataLoaderIter(_BaseDataLoaderIter):

    def __init__(self, loader, preprocess: Preprocess=None, queue_max_size=2):
        self._preprocess = preprocess
        if self._preprocess is not None and self._preprocess.has_cpu_preprocess():
            org_create_fetcher = _DatasetKind.create_fetcher
            _DatasetKind.create_fetcher = _create_fetcher_proxy(org_create_fetcher, self._preprocess)

        super(_PreprocessSingleProcessDataLoaderIter, self).__init__(loader)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

        if self._preprocess is not None and self._preprocess.has_cpu_preprocess():
            _DatasetKind.create_fetcher = org_create_fetcher

        self._timeout = self._timeout if self._timeout > 0 else (12 * _utils.MP_STATUS_CHECK_INTERVAL)

        if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
            self._stopped = False
            self._stream = torch.cuda.Stream()
            print(f"xxxxxx _PreprocessSingleProcessDataLoaderIter queue_max_size: {queue_max_size}")
            self._data_queue = queue.Queue(maxsize=queue_max_size)
            self._device_id = torch.cuda.current_device()
            self._preprocess_thread_done_event = threading.Event()
            self._preprocess_thread = threading.Thread(target=self._preprocess_loop)
            self._preprocess_thread.daemon = True
            self._preprocess_thread.start()

    def _preprocess_loop(self):
        # torch.set_num_threads(1)
        torch.cuda.set_device(self._device_id)
        while not self._preprocess_thread_done_event.is_set():
            while not self._stopped:
                index = None
                data = None
                try:
                    index = self._next_index()
                    data = self._dataset_fetcher.fetch(index)
                    if self._pin_memory:
                        data = _utils.pin_memory.pin_memory(data)
                    # print(f"_preprocess_loop data is {type(data)}")
                    # print(f"************** _PreprocessSingleProcessDataLoaderIter _preprocess_loop **************")
                    if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
                        with torch.no_grad(), torch.cuda.stream(self._stream):
                            data = self._preprocess.gpu_preprocess(data, self._stream)
                        # print(f"xxxxxx data: {data.keys()})")
                        self._stream.synchronize()
                        # print(f"xxxxxx _stream.synchronize")
                        # self._data_queue.put((index, data))
                except StopIteration as e:
                    # print("xxxxxx StopIteration")
                    self._stopped = True
                    index = None
                    data = None
                except Exception as e:
                    data = _utils.ExceptionWrapper(where=f"in _preprocess_loop thread for device {self._device_id}")
                r = (index, data)
                # print(f"xxxxxx r: ({index}, {data.keys()})")
                while not self._preprocess_thread_done_event.is_set():
                    try:
                        # print(f"xxxxxx will put data to queue, qsize: {self._data_queue.qsize()}")
                        self._data_queue.put(r, timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                        # print(f"xxxxxx success put data to queue")
                        break
                    except queue.Full:
                        # print(f"xxxxxx queue is full")
                        continue
                del r
                del index
                del data
            time.sleep(0.5)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._stopped = False

    def _next_data(self):
        data = None
        if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
            try:
                index, data = self._data_queue.get(timeout = self._timeout)
            except queue.Empty as e:
                raise e
            if isinstance(data, _utils.ExceptionWrapper):
                data.reraise()

            if index is None:
                self._shutdown_workers()
                raise StopIteration
            return data
        else:
            data = super()._next_data()
        return data

    def _shutdown_workers(self):
        if hasattr(self, '_preprocess_thread'):
            self._preprocess_thread_done_event.set()
            self._preprocess_thread.join()

    def __del__(self):
        self._shutdown_workers()


class _PreprocessMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):

    def __init__(self, loader, preprocess: Preprocess=None, queue_max_size=2):
        self._preprocess = preprocess
        if self._preprocess is not None and self._preprocess.has_cpu_preprocess():
            org_create_fetcher = _DatasetKind.create_fetcher
            _DatasetKind.create_fetcher = _create_fetcher_proxy(org_create_fetcher, self._preprocess)

        super(_PreprocessMultiProcessingDataLoaderIter, self).__init__(loader)

        if self._preprocess is not None and self._preprocess.has_cpu_preprocess():
            _DatasetKind.create_fetcher = org_create_fetcher

        # self._preprocess_out_queue = self._data_queue

        if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
            self._preprocess_in_queue = self._data_queue
            self._data_queue = queue.Queue(maxsize=queue_max_size)
            self._gpu_preprocess_thread_done_event = threading.Event()
            self._device_id = torch.cuda.current_device()
            self._timeout = self._timeout if self._timeout > 0 else (12 * _utils.MP_STATUS_CHECK_INTERVAL)
            self._stream = torch.cuda.Stream()

            self._gpu_preprocess_thread = threading.Thread(target=self._gpu_preprocess_loop)
            self._gpu_preprocess_thread.daemon = True
            self._gpu_preprocess_thread.start()


    def _gpu_preprocess_loop(self):
        torch.set_num_threads(1)
        # print("start _gpu_preprocess_loop")
        torch.cuda.set_device(self._device_id)
        while not self._gpu_preprocess_thread_done_event.is_set():
            try:
                r = self._preprocess_in_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                # print(f"r = {r}")
            except queue.Empty:
                # print(f"_preprocess_in_queue empty, timeout = {self._timeout}")
                continue
            idx, data = r
            # print(f"************** _PreprocessMultiProcessingDataLoaderIter _gpu_preprocess_loop **************")
            # print(f"_gpu_preprocess_loop data is {type(data)}")
            if not isinstance(idx, _utils.worker._ResumeIteration) and not self._gpu_preprocess_thread_done_event.is_set() and not isinstance(data, _utils.ExceptionWrapper):
                try:
                    if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
                        with torch.no_grad(), torch.cuda.stream(self._stream):
                            data = self._preprocess.gpu_preprocess(data, self._stream)
                                # print(f"data = {data}")
                except Exception as e:
                    # print(e)
                    data = _utils.ExceptionWrapper(where=f"in _gpu_preprocess_loop thread for device {self._device_id}")
                r = (idx, data)
            while not self._gpu_preprocess_thread_done_event.is_set():
                try:
                    self._data_queue.put(r, timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    # print(f"_data_queue full, timeout = {self._timeout}")
                    continue
            del r
            del idx
            del data
            
    def _shutdown_workers(self):
        super()._shutdown_workers()
        if hasattr(self, '_gpu_preprocess_thread'):
            self._gpu_preprocess_thread_done_event.set()
            self._gpu_preprocess_thread.join()


class PreprocessDataLoader(DataLoader):

    def __init__(self, preprocess: Preprocess=None, *args, **kwargs):
        super(PreprocessDataLoader, self).__init__(*args, **kwargs)
        print(f"init PreprocessDataLoader")
        self.preprocess = preprocess
        self.timeout=1800


    def _get_iterator(self) -> '_BaseDataLoaderIter':
        print(f"**************** PreprocessDataLoader _get_iterator ****************")
        if self.num_workers == 0:
            # print(f"will use  _PreprocessSingleProcessDataLoaderIter")
            return _PreprocessSingleProcessDataLoaderIter(self, self.preprocess)
        else:
            self.check_worker_number_rationality()
            return _PreprocessMultiProcessingDataLoaderIter(self, self.preprocess)



try:
    import ray
    from ray.util import queue as ray_queue
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from multiprocessing.managers import SyncManager

    old_deliver_challenge = mp.connection.deliver_challenge
    old_answer_challenge = mp.connection.answer_challenge

    def deliver_challenge_proxy(connection, authkey):
        # print(f"run deliver_challenge_proxy")
        old_deliver_challenge(connection, b"x")

    def answer_challenge_proxy(connection, authkey):
        # print(f"run answer_challenge_proxy")
        old_answer_challenge(connection, b"x")

    mp.connection.deliver_challenge = deliver_challenge_proxy
    mp.connection.answer_challenge = answer_challenge_proxy

    class QueueActor:
        def __init__(self, _index_queue, _result_queue):
            # print(f"queue1 = {type(queue1)}")
            self._index_queue = _index_queue
            self._result_queue = _result_queue

        def put_result(self, item, timeout=None):
            # print(f"****** begin to put item to queue1")
            self._result_queue.put(item, timeout=timeout)
            # print(f"****** success to put item to queue1")

        def get_index(self, timeout=None):
            return self._index_queue.get(timeout)

    class RpycActor(rpyc.Service):
        def __init__(self, _index_queue, _result_queue, ip, port):
            # print(f"queue1 = {type(queue1)}")
            self._index_queue = _index_queue
            self._result_queue = _result_queue
            self.ip = ip
            self.port = port

        def exposed_put_result(self, item, timeout=None):
            # print(f"****** begin to put item to queue1")
            # print("-----------QueueActor put_result start-----------", type(item))
            self._result_queue.put(item, timeout=timeout)
            # print("-----------QueueActor put_result end-----------", item[0])
            # print(f"****** success to put item to queue1")

        def exposed_get_index(self, timeout=None):
            return self._index_queue.get(timeout)
    @ray.remote(num_cpus=1)
    # @ray.remote
    class RayWorker(object):

        def __init__(self, _rank, _dataset_fetcher, _prefetch_factor):
            self._dataset_fetcher = _dataset_fetcher
            self._rank = _rank
            self._prefetch_factor = _prefetch_factor
            self._pool = ThreadPoolExecutor(max_workers=_prefetch_factor)

        def fetch(self, index):
            data = self._dataset_fetcher.fetch(index)
            return data
        
        def fetch_list(self, index_list, use_pool=True):
            if use_pool:
                data_list = self._pool.map(self.fetch, index_list)
                # data_list = [self._dataset_fetcher.fetch(index) for index in index_list]
                return list(data_list)
            else:
                return [self._dataset_fetcher.fetch(index) for index in index_list]
        
        def shutdown(self):
            self._pool.shutdown()


    @ray.remote(num_cpus=1)
    class RayQueueWorker(object):

        def __init__(self, _rank, _dataset_fetcher, _max_threads, _queue_actor, ip, port, _check_interval=12*_utils.MP_STATUS_CHECK_INTERVAL):
            self._dataset_fetcher = _dataset_fetcher
            self._rank = _rank
            self._queue_actor = _queue_actor
            self._check_interval = _check_interval
            self._worker_thread_done_event = threading.Event()
            self._max_threads = max(1, int(_max_threads))
            self._pool = ThreadPoolExecutor(max_workers=self._max_threads)
            self.conn = rpyc.connect(ip, port, config = rpyc_config)
            print(f"RayQueueWorker init, rank = {self._rank}")
        
        def fetch(self):
            while not self._worker_thread_done_event.is_set():
                try:
                    # print(f"RayQueueWorker begin fetch")
                    # index = ray.get(self._queue_actor.get_index.remote(timeout=self._check_interval))
                    index = self.conn.root.get_index(timeout=self._check_interval)
                    # print(f"get index = {index}")
                except queue.Empty:
                    # print(e)
                    # print(f"_preprocess_in_queue empty, timeout = {self._timeout}")
                    continue
                except Exception as e:
                    print("----get_index----", e)
                    time.sleep(10)
                    continue
                    # exit(1)
                try:
                    data = self._dataset_fetcher.fetch(index)
                    # print(f"success to load data")
                except Exception as e:
                    print("----_dataset_fetcher----", e)
                    data = _utils.ExceptionWrapper(where=f"in _worker_loop thread for device {self._device_id}")
                # print("get index and fetch data")
                r = (index, data)
                while not self._worker_thread_done_event.is_set():
                    try:
                        # ray.get(self._queue_actor.put_result.remote(item=r, timeout=self._check_interval))
                        self.conn.root.put_result(item=r, timeout=self._check_interval)
                        # print("success to put r")
                        break
                    except queue.Full:
                        continue
                    except Exception as e:
                        print("----put_result----", e)
                        time.sleep(10)
                        continue
                        # exit(1)
                del r
                del data
                del index
        
        def run(self):
            for i in range(self._max_threads):
                self._pool.submit(self.fetch)
        
        def shutdown(self):
            self.conn.close()
            self._worker_thread_done_event.set()
            self._pool.shutdown()

    class _RayPreprocessMultiProcessingDataLoaderIter(_BaseDataLoaderIter):

        def __init__(self, loader, preprocess: Preprocess=None, queue_max_size=10):
            self._preprocess = preprocess
            self._stopped = False
            self._send_idx = 0  # idx of the next task to be sent to workers
            self._rcvd_idx = 0  # idx of the next task to be returned in __next__

            super(_RayPreprocessMultiProcessingDataLoaderIter, self).__init__(loader)

            assert self._num_workers > 0
            assert self._prefetch_factor > 0



            # Adds forward compatibilities so classic DataLoader can work with DataPipes:
            #   Taking care of distributed sharding
            # if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            #     torch.utils.data.graph_settings.apply_sharding(self._dataset, self._world_size, self._rank)

            if self._preprocess is not None and self._preprocess.has_cpu_preprocess():
                org_create_fetcher = _DatasetKind.create_fetcher
                _DatasetKind.create_fetcher = _create_fetcher_proxy(org_create_fetcher, self._preprocess)
            self._dataset_fetcher = _DatasetKind.create_fetcher(
                self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)
            if self._preprocess is not None and self._preprocess.has_cpu_preprocess():
                _DatasetKind.create_fetcher = org_create_fetcher
            # if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
            self._queue_max_size = queue_max_size
            self._stream = torch.cuda.Stream()
            self._data_queue = queue.Queue(maxsize=self._queue_max_size)
            self._device_id = torch.cuda.current_device()
            self._timeout = self._timeout if self._timeout > 0 else (12 * _utils.MP_STATUS_CHECK_INTERVAL)
            print("init _RayPreprocessMultiProcessingDataLoaderIter")



        def _start(self):
            print("begin to start ray actor")

            if not ray.is_initialized():
                ray.init(address="auto")

            node_id = ray.get_runtime_context().get_node_id()
            scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            actor_options = {"num_cpus": 1, "scheduling_strategy": scheduling_strategy}
            # worker_actor_options = {"num_cpus": 1, "scheduling_strategy": scheduling_strategy}
            nodes = ray.nodes()
            node_cpu = 0
            node_gpu = 0
            for node in nodes:
                if node['NodeID'] == node_id:
                    node_cpu = int(node['Resources']['CPU'])
                    node_gpu = int(node['Resources']['GPU'])
                    break
            
            num_queue_actors = node_cpu // int(max(1, node_gpu))
            print(f"node_id = {node_id}, node_cpu = {node_cpu}, node_gpu = {node_gpu}, num_queue_actors = {num_queue_actors}")
            
            self._index_thread_done_event = threading.Event()
            # self._index_queue = ray_queue.Queue(maxsize=queue_max_size, actor_options=actor_options)
            self._manager = multiprocessing.Manager()
            
            self._index_queue = self._manager.Queue(maxsize=self._queue_max_size)
            self._worker_result_queue = self._manager.Queue(maxsize=2000)
            self._p_queue_actor_list = []
            self._t_queue_actor_list = []
            self._queue_actor_list = []
            port = 18861
            ip = get_address_ip()
            for i in range(num_queue_actors):
                # _queue_actor = ray.remote(QueueActor).options(**actor_options).remote(self._index_queue, self._worker_result_queue)
                t_port = port + i

                _queue_actor = RpycActor(self._index_queue, self._worker_result_queue, ip, t_port)
                self._queue_actor_list.append(_queue_actor)

                t = ThreadedServer(_queue_actor, port=t_port, protocol_config=rpyc_config)
                self._t_queue_actor_list.append(t)

                p = Process(target=queue_actor_run, args=(t,))
                self._p_queue_actor_list.append(p)
                p.start()

            self._index_thread = threading.Thread(target=self._index_loop)
            self._index_thread.daemon = True
            self._index_thread.start()

            # self._worker_thread_done_event = threading.Event()
            # self._worker_result_queue = ray_queue.Queue(maxsize=queue_max_size, actor_options=actor_options)
            # self._worker_result_queue = self._manager.Queue(maxsize=self._queue_max_size)
            # self._worker_result_queue_actor = ray.remote(QueueActor).options(**actor_options).remote()
            # self._worker_result_queue_actor.set_queue.remote(self._worker_result_queue, self._queue_max_size)
            self._worker_actor_list = []
            # print("****** init _worker_actor_list")
            for i in range(self._num_workers):
                _queue_actor = self._queue_actor_list[i % num_queue_actors]
                # _worker_actor = ray.remote(RayQueueWorker).options(**actor_options).remote(i, self._dataset_fetcher, self._prefetch_factor, self._index_queue_actor, self._worker_result_queue_actor, self._timeout)
                # _worker_actor = RayQueueWorker.remote(i, self._dataset_fetcher, self._prefetch_factor, _queue_actor)
                _worker_actor = RayQueueWorker.remote(i, self._dataset_fetcher, self._prefetch_factor, None, _queue_actor.ip, _queue_actor.port)
                _worker_actor.run.remote()
                self._worker_actor_list.append(_worker_actor)

            self._gpu_preprocess_thread_done_event = threading.Event()
            self._gpu_preprocess_thread = threading.Thread(target=self._gpu_preprocess_loop)
            self._gpu_preprocess_thread.daemon = True
            self._gpu_preprocess_thread.start()
            print("success to start ray actor")
            # self._reset(loader, first_iter=True)
        # def _index_loop(self):
        #     torch.set_num_threads(1)

        #     index_list_list = []
        #     worker_result_ref_list = []

        #     while not self._index_thread_done_event.is_set():
        #         while not self._stopped:

        #             index_list_list.clear()
        #             worker_result_ref_list.clear()
        #             start_time = time.time()
        #             for i in range(len(self._worker_actor_list)):
        #                 index_list = []
        #                 for j in range(self._prefetch_factor):
        #                     try:
        #                         index_list.append(self._next_index())
        #                     except StopIteration as e:
        #                         self._stopped = True
        #                         break
        #                 if len(index_list) > 0:
        #                     index_list_list.append(index_list)
        #             end_time = time.time()
        #             index_time = end_time - start_time

        #             start_time = time.time()
        #             if not self._index_thread_done_event.is_set():
        #                 try:
        #                     for j in range(len(index_list_list)):
        #                         worker = self._worker_actor_list[j]
        #                         index_list = index_list_list[j]
        #                         worker_result_ref_list.append(worker.fetch_list.remote(index_list))
        #                     worker_result_list = ray.get(worker_result_ref_list)

        #                     r_list = []
        #                     for (index_list, data_list) in zip(index_list_list, worker_result_list):
        #                         for (index, data) in zip(index_list, data_list):
        #                             r_list.append((index, data))
        #                     # r_list = [(index, data) for (index, data) in zip(index_list, worker_result_list)]
        #                     del worker_result_list
                            
        #                 except Exception as e:
        #                     # print(f"len(index_list_list) = {len(index_list_list)}, len(self._worker_actor_list) = {len(self._worker_actor_list)}")
        #                     data = _utils.ExceptionWrapper(where=f"in _worker_loop thread for device {self._device_id}")
        #                     r_list = []
        #                     for index_list in index_list_list:
        #                         for index in index_list:
        #                             r_list.append((index, data))
        #                     print(e)
        #                 end_time = time.time()
        #                 ray_time = end_time - start_time

        #                 start_time = time.time()
        #                 for r in r_list:
        #                     while not self._index_thread_done_event.is_set():
        #                         try:
        #                             self._worker_result_queue.put(r, timeout=self._timeout)
        #                             break
        #                         except queue.Full:
        #                             continue

        #                 while self._stopped and not self._index_thread_done_event.is_set():
        #                     try:
        #                         self._worker_result_queue.put((None, None), timeout=self._timeout)
        #                         break
        #                     except queue.Full:
        #                         continue
        #                 del r_list
        #                 end_time = time.time()
        #                 put_time = end_time - start_time
        #                 # print(f"index_time = {index_time} sec, ray_time = {ray_time} sec, put_time = {put_time} sec")

        #         time.sleep(1)
            
        #     del index_list_list
        #     del worker_result_ref_list

        def _reset(self, loader, first_iter=False):
            super()._reset(loader, first_iter)
            self._send_idx = 0
            self._rcvd_idx = 0
            self._stopped = False

        def _index_loop(self):
            torch.set_num_threads(1)
            while not self._index_thread_done_event.is_set():
                while not self._stopped:
                    try:
                        index = self._next_index()
                        # print(f"r = {r}")
                    except StopIteration as e:
                        self._stopped = True
                        break
                    while not self._index_thread_done_event.is_set():
                        try:
                            self._index_queue.put(index, timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                            self._send_idx += 1
                            # print("put index")
                            break
                        except queue.Full:
                            continue
                    del index

                time.sleep(1)


        def _gpu_preprocess_loop(self):
            torch.set_num_threads(1)
            torch.cuda.set_device(self._device_id)
            total_time = 0
            
            num = 0
            # queue_size = 0
            while not self._gpu_preprocess_thread_done_event.is_set():
                start_time = time.time()
                try:
                    r = self._worker_result_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    self._rcvd_idx += 1
                except queue.Empty:
                    if not self._stopped:
                        # print("queue empty")
                        # print(f"_preprocess_in_queue empty, timeout = {self._timeout}")
                        continue
                    else:
                        # r = (None, None)
                        if self._rcvd_idx < self._send_idx:
                            continue
                        else:
                            print("----ray success----", self._send_idx, self._rcvd_idx)
                            break
                except Exception as e:
                    print(e)
                    exit(1)


                # try:
                #     # queue_size = self._worker_result_queue.size()
                #     queue_size = int(max(self._queue_max_size - 2, 1))
                #     r_list = self._worker_result_queue.get_nowait_batch(queue_size)
                # except queue.Empty:
                #     try:
                #         r_list = []
                #         r = self._worker_result_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                #         r_list.append(r)
                #     except queue.Empty:
                #         if not self._stopped:
                #             print("queue empty")
                #             # print(f"_preprocess_in_queue empty, timeout = {self._timeout}")
                #             continue
                #         else:
                #             r_list = []
                #             r = (None, None)
                #             r_list.append(r)

                # num += 1
                # total_time += (time.time() - start_time)
                # avg_speed = num / total_time
                # avg_time = total_time / num
                # print(f"avg_speed = {avg_speed} batch/s, avg_time = {avg_time} s/batch, qsize = {self._worker_result_queue.qsize()}")

                try:
                    index, data = r
                    if index is not None and not isinstance(data, _utils.ExceptionWrapper):
                        if self._pin_memory:
                            data = _utils.pin_memory.pin_memory(data)
    
                        # print(f"_preprocess_loop data is {type(data)}")
                        if self._preprocess is not None and self._preprocess.has_gpu_preprocess():
                        
                            with torch.no_grad(), torch.cuda.stream(self._stream):
                                data = self._preprocess.gpu_preprocess(data, self._stream)
                except Exception as e:
                    data = _utils.ExceptionWrapper(where=f"in _gpu_preprocess_loop thread for device {self._device_id}")
                    print(e)
    
                r = (index, data)
                while not self._gpu_preprocess_thread_done_event.is_set():
                    try:
                        self._data_queue.put(r, timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                        # print("_gpu_preprocess_loop put r")
                        break
                    except queue.Full:
                        continue
                    except Exception as e:
                        print(e)
                        exit(1)
                del r
                del index
                del data


        def _next_data(self):
            if self._rcvd_idx == self._send_idx and self._send_idx > 0:
                print("-----Stop:send_idx,rcvd_idx-----", self._send_idx, self._rcvd_idx)
                for _t in self._t_queue_actor_list:
                    _t.close()
                for _p in self._p_queue_actor_list:
                    _p.join()
                    _p.close()
                raise StopIteration
            try:
                index, data = self._data_queue.get(timeout = self._timeout)
            except queue.Empty as e:
                raise e
            if isinstance(data, _utils.ExceptionWrapper):
                data.reraise()

            if index is None:
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration
            return data


        def _shutdown_workers(self):
            print("begin to shutdown ray actors")
            self._index_thread_done_event.set()
            self._index_thread.join()

            # self._worker_thread_done_event.set()
            # for _worker_thread in self._worker_thread_list:
            #     _worker_thread.join()
            # self._worker_thread_list.clear()

            self._gpu_preprocess_thread_done_event.set()
            self._gpu_preprocess_thread.join()

            shutdown_task = []
            for _worker in self._worker_actor_list:
                shutdown_task.append(_worker.shutdown.remote())
            ray.wait(shutdown_task)
            for _worker in self._worker_actor_list:
                ray.kill(_worker, no_restart=True)
            self._worker_actor_list.clear()

            # for _queue in self._queue_actor_list:
                # ray.kill(_queue, no_restart=True)

            # ray.kill(self._index_queue_actor, no_restart=True)
            # ray.kill(self._worker_result_queue_actor, no_restart=True)
            self._manager.shutdown()
            print("success to shutdown ray actors")



        def __del__(self):
            self._shutdown_workers()


    class RayPreprocessDataLoader(PreprocessDataLoader):

        def __init__(self, preprocess: Preprocess=None, *args, **kwargs):
            self.queue_max_size = int(kwargs.pop('queue_max_size', 10))
            kwargs['persistent_workers'] = True
            super(RayPreprocessDataLoader, self).__init__(preprocess, *args, **kwargs)
            

        def _get_iterator(self) -> '_BaseDataLoaderIter':
            if self.num_workers == 0:
                return _PreprocessSingleProcessDataLoaderIter(self, self.preprocess)
            else:
                # self.check_worker_number_rationality()
                ray_iter = _RayPreprocessMultiProcessingDataLoaderIter(self, self.preprocess, queue_max_size=self.queue_max_size)
                ray_iter._start()
                return ray_iter

except Exception as e:
    print(e)
    print(f"Can not find ray! if you want to use RayPreprocessDataLoader, please run: \n"
        + "\t pip install gpustat-1.0.0.tar.gz -i https://pypi.mirrors.ustc.edu.cn/simple/; \n"
        + "\t pip install 'ray[default]' -i https://pypi.mirrors.ustc.edu.cn/simple/;")

if __name__== "__main__":
    import time
    from tqdm import tqdm
    import lmdb
    from staracc_cv.data.dataset import LMDBDataSet
    import os

    os.environ['RAY_memory_usage_threshold'] = '0'

  